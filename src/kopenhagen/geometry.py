"""Plan view of the stair head, reconstructed from photos IMG_3394-3399.

Coordinates in metres. The stair runs along y and rises towards +y; the top
nosing sits at y = 0. The upper floor (landing + gallery) is y in [0, depth];
the gallery extends to the left (-x) of the stair. The right glass balustrade
runs along x = +W/2 from the stair foot past the column and ends at
y = glass_end, where it turns right along the void; above that line the floor
opens to the right. The observed path (user sketch) comes in from the right
through that opening, turns left around the glass end, passes the column on its
wide left side and turns back onto the stair. The left
glass balustrade ends at the top nosing, so its end point (-W/2, 0) is the
inner corner that people turn around when they come from the gallery.

All dimensions are estimates from photos (tread depth ~0.3 m as scale).
"""

from __future__ import annotations

from dataclasses import dataclass, replace

from shapely import LineString, Point, Polygon, box
from shapely.ops import unary_union


@dataclass(frozen=True)
class Layout:
    stair_width: float = 2.0  # glass to glass
    stair_length: float = 4.0  # plan length modelled below the nosing
    floor_depth: float = 5.0  # nosing to the far end of the modelled upper floor
    gallery_length: float = 6.0  # gallery left of the stair
    column_diameter: float = 0.5
    column_setback: float = 1.0  # nosing to column centre
    column_offset: float = 0.1  # column centre right of the stair axis
    glass_end: float = 2.6  # right glass ends here and turns right
    right_floor_length: float = 3.5  # floor opening to the right above glass_end
    has_column: bool = True

    def with_(self, **kw) -> Layout:
        return replace(self, **kw)

    @property
    def x_left(self) -> float:
        return -self.stair_width / 2

    @property
    def x_right(self) -> float:
        return self.stair_width / 2

    @property
    def x_gallery_end(self) -> float:
        return self.x_left - self.gallery_length

    @property
    def inner_corner(self) -> tuple[float, float]:
        return (self.x_left, 0.0)


def column(lay: Layout) -> Polygon | None:
    if not lay.has_column:
        return None
    centre = Point(lay.column_offset, lay.column_setback)
    return centre.buffer(lay.column_diameter / 2, quad_segs=16)


def walkable_area(lay: Layout) -> Polygon:
    upper = box(lay.x_gallery_end, 0.0, lay.x_right, lay.floor_depth)
    stair = box(lay.x_left, -lay.stair_length, lay.x_right, 0.0)
    right = box(lay.x_right, lay.glass_end, lay.x_right + lay.right_floor_length, lay.floor_depth)
    area = unary_union([upper, stair, right])
    col = column(lay)
    if col is not None:
        area = area.difference(col)
    return area


def glass(lay: Layout) -> dict[str, LineString]:
    foot = -lay.stair_length
    return {
        "left": LineString([(lay.x_left, foot), lay.inner_corner]),
        "right": LineString([
            (lay.x_right, foot),
            (lay.x_right, lay.glass_end),
            (lay.x_right + lay.right_floor_length, lay.glass_end),
        ]),
    }


def gaps(lay: Layout) -> dict[str, float]:
    """Clear widths around the column (model-independent check)."""
    if not lay.has_column:
        return {}
    r = lay.column_diameter / 2
    cx, cy = lay.column_offset, lay.column_setback
    to_right_glass = lay.x_right - (cx + r)
    to_corner = Point(lay.inner_corner).distance(Point(cx, cy)) - r
    return {
        "column_to_right_glass": to_right_glass,
        "column_to_inner_corner": to_corner,
        "column_to_nosing": cy - r,
    }


def column_bypass(lay: Layout) -> tuple[float, float] | None:
    """Waypoint on the wide left side of the column, as in the observed path."""
    if not lay.has_column:
        return None
    r = lay.column_diameter / 2
    return (lay.column_offset - r - 0.45, lay.column_setback)


def lane_waypoints(lay: Layout, direction: str) -> list[tuple[float, float]]:
    """Keep-right checkpoints: on the stair, down on the -x half and up on the +x half.

    People coming up keep right past the column, through the gap to the right glass, and
    round the end of the right glass on its near side; people going down pass it on the far side.
    """
    lane = lay.stair_width / 4
    foot = -lay.stair_length
    if direction == "up_left" and lay.has_column:  # too wide for the gap: round the column's left side
        bx, by = column_bypass(lay)
        return [(lane, foot + 1.0), (lane, -0.4), (bx, by), (bx + 0.2, by + 1.0),
                (lay.x_right + 0.4, lay.glass_end + 0.45)]
    if direction in ("up_right", "up_left"):
        gap_x = (lay.column_offset + lay.column_diameter / 2 + lay.x_right) / 2
        return [(lane, foot + 1.0), (lane, -0.4), (gap_x, lay.column_setback),
                (lay.x_right + 0.4, lay.glass_end + 0.45)]
    return [(-lane, lay.column_setback), (-lane, -0.8), (-lane, foot + 1.0)]


def far_side_waypoint(lay: Layout) -> tuple[float, float]:
    """Where people going down round the end of the right glass when keeping right."""
    return (lay.x_right + 0.4, min(lay.glass_end + 1.4, lay.floor_depth - 0.4))


def spawn_zone(lay: Layout, direction: str, centred: bool = False, wide: bool = False,
               keep_right: bool = False) -> Polygon:
    top = lay.floor_depth
    x1 = lay.x_right + lay.right_floor_length
    if direction == "right" and wide:
        return box(x1 - 2.5, lay.glass_end + 0.3, x1 - 0.3, top - 0.3)
    if direction == "right":
        y1 = min(lay.glass_end + 1.4, top - 0.3)
        return box(x1 - 1.5, lay.glass_end + 0.3, x1 - 0.3, y1)
    if direction == "down" and centred:
        return box(-0.7, top - 1.5, 0.7, top - 0.3)
    if direction == "down":
        return box(lay.x_left - 1.0, top - 1.5, lay.x_right - 0.3, top - 0.3)
    if direction == "gallery":
        x0 = lay.x_gallery_end
        return box(x0 + 0.3, 0.3, x0 + 2.3, 2.3)
    foot = -lay.stair_length
    if keep_right:
        return box(0.15, foot + 0.3, lay.x_right - 0.3, foot + 1.5)
    return box(lay.x_left + 0.3, foot + 0.3, lay.x_right - 0.3, foot + 1.5)


def exit_zone(lay: Layout, direction: str) -> Polygon:
    if direction == "up_right":
        x1 = lay.x_right + lay.right_floor_length
        return box(x1 - 0.6, lay.glass_end, x1, lay.floor_depth)
    if direction in ("down", "gallery", "right"):
        foot = -lay.stair_length
        return box(lay.x_left, foot, lay.x_right, foot + 0.6)
    top = lay.floor_depth
    return box(lay.x_left - 1.0, top - 0.6, lay.x_right, top)


SCENARIOS: dict[str, Layout] = {
    "as built": Layout(),
    "no column": Layout(has_column=False),
    "column set back 2 m": Layout(column_setback=2.0),
    "column against right glass": Layout(column_offset=0.75),
}
