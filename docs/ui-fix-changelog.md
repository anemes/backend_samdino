# UI Fix: Annotations Not Appearing + Polygon Tool + Architecture Cleanup

**Date**: 2026-02-16

---

## Root Causes Found

### Bug 1: SAM3 Annotations in Pixel Coordinates (CRITICAL)

**File**: `backend/hitl/services/sam_service.py` — `mask_to_polygon()`

When the plugin called `/api/sam/accept`, it never sent an affine transform. The backend defaulted to `Affine.identity()`, so the vectorized polygon had pixel coordinates (e.g., `(100, 200)`) instead of geo-coordinates. These were stored in the GeoPackage as if they were EPSG:3857 meters, placing them near 0°N, 0°E — invisible at the user's actual map location.

### Bug 2: Polygon Tool — Silent Containment Check Failure

**File**: `backend/hitl/api/labels.py` — `add_annotation()`

The annotation endpoint validates that the annotation centroid is inside the target region. If the wrong region was selected (or `get_active_region_id()` silently defaulted to `1`), the 400 error was raised but easily missed in the QGIS message bar.

### Bug 3: SAM3 Accept Bypassed Containment Check

**File**: `backend/hitl/api/sam.py` — `accept_mask()`

The SAM accept endpoint called `store.add_annotation()` directly, skipping the containment check that the manual annotation endpoint enforced.

### Bug 4: Fragmented Sync Flow

Signal wiring was scattered — some handlers called `refresh_regions()` + `layers_changed.emit()`, some only called one, leading to inconsistent UI updates after mutations.

---

## Changes Made

### Backend

#### `backend/hitl/services/sam_service.py`
- **`mask_to_polygon()`**: Removed the `transform` parameter. Now reads the affine transform AND CRS directly from the session's GeoTIFF using `rasterio.open()`. If the target CRS differs from the image CRS, reprojects the polygon using pyproj.
- This was the critical fix — polygons now have correct geo-coordinates.

#### `backend/hitl/api/sam.py`
- Removed `transform` from `AcceptRequest` pydantic model.
- `accept_mask()`: Added `store.check_annotation_in_region()` validation before saving, matching the same check used by `/api/labels/annotations`.

### Plugin

#### `qgis_plugin/hitl_sketcher/connection/client.py`
- `sam_accept()`: Removed unused `transform` parameter.

#### `qgis_plugin/hitl_sketcher/labeling/project_panel.py`
- **`_on_accept()`**: Uses `crs="EPSG:4326"` consistently (backend handles reprojection from image CRS). Checks for `None` region before sending. Shows class name + region ID in success message.
- **`get_active_region_id()`**: Returns `Optional[int]` — `None` when no regions exist instead of silently defaulting to `1`. Auto-selects first region if none highlighted.
- **Added status label**: Shows active class + region above tool buttons (e.g., "Class: 2: building | Region 1"). Turns red italic when no region exists.
- **Removed redundant `refresh_regions()` calls** from `_on_delete_region`, `_on_delete_annotation`, `_on_clear_region_annotations`, `_on_accept` — sync is now consolidated in `plugin.py`.

#### `qgis_plugin/hitl_sketcher/labeling/polygon_tool.py`
- **`_finalize()`**: Checks for `None` region before attempting to save. Distinguishes "outside region" errors from other failures with specific error messages.

#### `qgis_plugin/hitl_sketcher/plugin.py`
- **Consolidated sync**: Replaced `_on_region_created`, `_on_annotation_saved`, `_sync_layers` with single `_sync_all()` that refreshes both QGIS layers AND panel region list.
- All mutation signals (`region_created`, `annotation_saved`, `mask_accepted`, `layers_changed`) point to `_sync_all()`.
- **Tool button state**: Added `_on_map_tool_changed()` connected to `canvas.mapToolSet` — unchecks panel tool buttons when user switches to pan/zoom/other tools.

---

## Data Flow (After Fix)

### SAM3 Accept Flow
```
User clicks on map → SAMTool._handle_click()
  → client.sam_prompt(pixel_coords) → backend returns mask PNG
  → MaskOverlay displays mask on canvas

User clicks "Accept Mask" → ProjectPanel._on_accept()
  → client.sam_accept(class_id, region_id, crs="EPSG:4326")
  → Backend: sam_service.mask_to_polygon()
    → reads affine transform from session GeoTIFF (rasterio)
    → vectorizes mask with real geo-transform
    → reprojects from image CRS to EPSG:4326 if needed
  → Backend: store.check_annotation_in_region() (containment check)
  → Backend: store.add_annotation() (saves to GeoPackage)
  → Plugin: layers_changed.emit() → _sync_all()
    → label_manager.sync_all() (rebuilds QGIS layers from backend)
    → project_panel.refresh_regions() (updates region list + counts)
```

### Polygon Tool Flow
```
User clicks "Draw Polygon" → ProjectPanel._on_polygon_tool()
  → polygon_tool_requested.emit() → plugin._activate_polygon_tool()
  → canvas.setMapTool(polygon_tool)

User draws polygon (left-click vertices, right-click finish)
  → PolygonTool._finalize()
  → Checks region_id is not None
  → client.add_annotation(geojson, class_id, region_id, crs=canvas_crs)
  → Backend: containment check + save to GeoPackage
  → annotation_saved.emit() → _sync_all()
```

### Consolidated Sync
```
Any mutation signal → _sync_all()
  → label_manager.sync_all()     # rebuild QGIS in-memory layers
  → project_panel.refresh_regions()  # update panel region list + annotation counts
```
