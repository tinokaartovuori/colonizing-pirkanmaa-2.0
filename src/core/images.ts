// Port of Graphics/imagevectors.h and Graphics/animationoptions.h.
// Original paths look like ":Images/grassland.png"; the Phaser texture key is
// just the base filename without extension ("grassland").

export type ImageVector = string[];

function key(path: string): string {
  // ":Images/grassland.png" -> "grassland"
  const file = path.substring(path.lastIndexOf('/') + 1);
  return file.replace(/\.png$/, '');
}
function vec(...paths: string[]): ImageVector {
  return paths.map(key);
}

export const ImageVectors = {
  CLICKEDTILEBORDER: vec(':Images/selectionborder.png'),
  MOUSEHOVERBORDER: vec(':Images/tilemousehover_1.png', ':Images/tilemousehover_2.png'),
  TILEOWNERBORDERS: vec(
    ':Images/playeroneborder_n.png',
    ':Images/playertwoborder_n.png',
    ':Images/playerthreeborder_n.png',
    ':Images/playerfourborder_n.png',
  ),
  FOREST_1: vec(':Images/forest_1_1.png', ':Images/forest_1_2.png', ':Images/forest_1_3.png'),
  FOREST_2: vec(':Images/forest_2_1.png', ':Images/forest_2_2.png', ':Images/forest_2_3.png'),
  FOREST_STUMPS: vec(':Images/foreststumps.png'),
  GRASSLAND: vec(':Images/grassland.png'),
  MIKONTALO: vec(':Images/mikontalo.png'),
  ABUNDANT_FOREST: vec(
    ':Images/abundant_forest_1.png',
    ':Images/abundant_forest_2.png',
    ':Images/abundant_forest_3.png',
  ),
  MOUNTAIN: vec(':Images/mountain.png'),
  MOUNTAIN_FOREST: vec(':Images/mountain_f_1.png', ':Images/mountain_f_2.png', ':Images/mountain_f_3.png'),
  HEADQUARTERSONE: vec(
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayerone2.png',
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayerone4.png',
  ),
  HEADQUARTERSTWO: vec(
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayertwo2.png',
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayertwo4.png',
  ),
  HEADQUARTERSTHREE: vec(
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayerthree2.png',
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayerthree4.png',
  ),
  HEADQUARTERSFOUR: vec(
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayerfour2.png',
    ':Images/headquarters1_3.png',
    ':Images/headquartersplayerfour4.png',
  ),
  HEADQUARTERSDESTROYED: vec(':Images/headquartersDestroyed.png'),
  OUTPOST: vec(':Images/outpost_1.png', ':Images/outpost_2.png', ':Images/outpost_3.png'),
  HYDROPOWERNS: vec(':Images/hydropower1NS.png', ':Images/hydropower2NS.png'),
  HYDROPOWERWE: vec(':Images/hydropower1WE.png', ':Images/hydropower2WE.png'),
  NUCLEARPLANT: vec(':Images/nuclearPlant1.png', ':Images/nuclearPlant2.png'),
  VILLAGE: vec(':Images/village.png'),
  BRIDGENS: vec(':Images/bridgeNS.png'),
  BRIDGEWE: vec(':Images/bridgeWE.png'),
  MINE: vec(':Images/mine.png'),
  FARM: vec(':Images/farm1.png', ':Images/farm2.png', ':Images/farm3.png', ':Images/farm4.png'),
  BASICWORKER: vec(':Images/basicworker_1.png', ':Images/basicworker_2.png'),
  EXPERT: vec(':Images/expert_1.png', ':Images/expert_2.png'),
  SOLDIER: vec(':Images/soldier_1.png', ':Images/soldier_2.png'),
  BASICWORKER_SWIM: vec(':Images/basicworker_swim_1.png', ':Images/basicworker_swim_2.png'),
  EXPERT_SWIM: vec(':Images/expert_swim_1.png', ':Images/expert_swim_2.png'),
  SOLDIER_SWIM: vec(':Images/soldier_swim_1.png', ':Images/soldier_swim_2.png'),
  COVER_BORDER: vec(':Images/tile_cover_border.png'),
  MENU: vec(':Images/menu_bg.png'),
  CONTAINER: vec(':Images/container_2_2.png'),
  BUTTON: vec(':Images/button_1_2.png'),
  RED: vec(':Images/red.png'),
  BLUE: vec(':Images/blue.png'),
  PURPLE: vec(':Images/purple.png'),
  YELLOW: vec(':Images/yellow.png'),
  BAR_RED: vec(':Images/color_bar_red.png'),
  BAR_BLUE: vec(':Images/color_bar_blue.png'),
  BAR_PURPLE: vec(':Images/color_bar_purple.png'),
  BAR_YELLOW: vec(':Images/color_bar_yellow.png'),
  BAR_NEUTRAL: vec(':Images/color_bar_neutral.png'),
  BLOCKED_TILE: vec(':Images/blocked_tile.png'),
  MONEY: vec(':Images/money.png'),
  WOOD: vec(':Images/wood.png'),
  STONE: vec(':Images/stone.png'),
  METAL: vec(':Images/metal.png'),
  MULTI: vec(
    ':Images/multi_0.png',
    ':Images/multi_1.png',
    ':Images/multi_2.png',
    ':Images/multi_3.png',
    ':Images/multi_4.png',
    ':Images/multi_5.png',
    ':Images/multi_6.png',
    ':Images/multi_7.png',
    ':Images/multi_8.png',
  ),
  RIVER_EW: vec(':Images/river_ew_1.png', ':Images/river_ew_2.png'),
  RIVER_NS: vec(':Images/river_ns_1.png', ':Images/river_ns_2.png'),
  RIVER_NW: vec(':Images/river_nw_1.png', ':Images/river_nw_2.png'),
  RIVER_NE: vec(':Images/river_ne_1.png', ':Images/river_ne_2.png'),
  RIVER_SW: vec(':Images/river_sw_1.png', ':Images/river_sw_2.png'),
  RIVER_SE: vec(':Images/river_se_1.png', ':Images/river_se_2.png'),
} as const;

// ---------------------------------------------------------------------------
// AnimationOption (Graphics/animationoption.{h,cpp})
// ---------------------------------------------------------------------------

export type AnimStyle = 'rollover' | 'backandforth';

export class AnimationOption {
  readonly animated: boolean;
  readonly style: AnimStyle;
  readonly randomFrame: boolean;

  constructor(onoff = false, style: AnimStyle = 'rollover', randomFrame = false) {
    this.animated = onoff;
    this.style = style;
    this.randomFrame = randomFrame;
  }
}

export const AnimationOptions = {
  CLICKEDTILEBORDER: new AnimationOption(false),
  MOUSEHOVERBORDER: new AnimationOption(true, 'rollover'),
  FOREST: new AnimationOption(true, 'backandforth', true),
  GRASSLAND: new AnimationOption(false),
  MIKONTALO: new AnimationOption(false),
  MOUNTAIN: new AnimationOption(false),
  MOUNTAIN_FOREST: new AnimationOption(true, 'backandforth', true),
  CONTAINER: new AnimationOption(false),
  BUTTON: new AnimationOption(false),
  MENU: new AnimationOption(false),
  HEADQUARTERS: new AnimationOption(true, 'rollover'),
  UNIT: new AnimationOption(true, 'rollover'),
  NUCLEAR: new AnimationOption(true, 'rollover'),
  HEPP: new AnimationOption(true, 'backandforth'),
  OUTPOST: new AnimationOption(true, 'backandforth'),
  EMPTY: new AnimationOption(false),
  RIVER: new AnimationOption(true, 'rollover', true),
} as const;

/** Every unique texture key the game uses, for preloading. */
export function allTextureKeys(): string[] {
  const keys = new Set<string>();
  for (const v of Object.values(ImageVectors)) {
    for (const k of v) keys.add(k);
  }
  return [...keys];
}
