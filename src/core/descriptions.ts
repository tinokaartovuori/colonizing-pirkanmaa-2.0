// Port of Core/descriptionmaps.h (Student::ConstDescriptionMaps).

import { FARM_GROW_TIME, VILLAGE_UNIT_VALUE, OUTPOST_SOLDIER_VALUE, MIKONTALO_UNIT_VALUE } from './resources';

export const FARM_DESCRIPTION =
  'This is a lovely place. Crops can be grown here. ' +
  'Never leave the crops alone or they will die! ' +
  'Crops can be harvested after <u>' +
  FARM_GROW_TIME +
  ' rounds</u>.';

export const BRIDGE_DESCRIPTION =
  'You can use this to cross rivers. Bridge needs a little maintenance every round.';

export const VILLAGE_DESCRIPTION =
  'Increases the amount of units you can have by <u>' +
  VILLAGE_UNIT_VALUE +
  '</u>! For soldiers you will need something else...';

export const HEPP_DESCRIPTION =
  'Uses water flow to produce energy. This hydroelectric power plant is ' +
  'kinda advanced stuff so you will need at least one expert here.';

export const NUCLEAR_DESCRIPTION =
  'Nuclear power plant is the most efficient power plant available. ' +
  'This is very dangerous and advanced technology so at least one expert ' +
  'is required here.';

export const MINE_DESCRIPTION =
  'Mine is a very fun place to work. You can be in the dark whole day and' +
  ' mine some stone and metal. Expert can make your work much better though.';

export const OUTPOST_DESCRIPTION =
  'Good place for soldiers to hang out. Increases the amount of soldiers you' +
  ' can have by <u>' +
  OUTPOST_SOLDIER_VALUE +
  '</u>! Enemy cannot directly attack this building.';

export const HEADQUARTERS_DESCRIPTION =
  'This is the heart of your region. If enemy gets here you lose. ' +
  'Because of the ultimate security no units allowed here.';

export const BROKEN_HEADQUARTERS_DESCRIPTION = 'Headquarters that got destroyed...';

export const FOREST_DESCRIPTION =
  'Forest is the only way to get wood. ' +
  'More workers make chopping down forest faster. ' +
  'Forest will grow back but you can build on top of it after cutting.';

export const GRASSLAND_DESCRIPTION =
  'Just a bunch of grass and stuff... There is a lot of stuff that can be built here.';

export const MIKONTALO_DESCRIPTION =
  'Some big house that seems to be founded in 1978-1980. ' +
  'Owning this awesome place increases the amount of units you can have' +
  ' by <u>' +
  MIKONTALO_UNIT_VALUE +
  '</u>. No need for maintenance either.';

export const MOUNTAIN_DESCRIPTION =
  'I love high places. Maybe there is something valuable inside this big rock thingy... ' +
  'Consider building a mine here.';

export const RIVER_DESCRIPTION_1 =
  'I like how it flows... Maybe there is a way to get over it... ' +
  'Or maybe the flow can be used to generate power.';

export const RIVER_DESCRIPTION_2 =
  'Seems to be a bit too curvy place for a bridge... The flow is not that great either.';

export const ABUNDANT_FOREST_DESCRIPTION = 'A lush forest where you can forage some juicy fruits.';

export const FARM_SHOP_DESCRIPTION =
  'Worker can grow crops. Crops can be harvested every <u>' +
  FARM_GROW_TIME +
  '</u> rounds. Never leave the crops alone!';

export const BRIDGE_SHOP_DESCRIPTION = 'You can use this to cross some rivers! Remember to do the maintenace.';

export const VILLAGE_SHOP_DESCRIPTION =
  'Increases the amount of units you can have by <u>' +
  VILLAGE_UNIT_VALUE +
  '</u>! Soldiers will need something else though.';

export const HEPP_SHOP_DESCRIPTION = 'Produces a lot of money! This power plant requires at least one expert.';

export const NUCLEAR_SHOP_DESCRIPTION = 'Produces a ton of money! You will need at least one expert to do so.';

export const MINE_SHOP_DESCRIPTION = 'You can mine some stone and metal! An expert can increase the efficiency.';

export const OUTPOST_SHOP_DESCRIPTION =
  'Increases the amount of soldiers you can have by <u>' +
  OUTPOST_SOLDIER_VALUE +
  '</u>. Also protects tile directly next to it.';

export const STRANGE_DEVICE_DESCRIPTION =
  'A strange humming device of unknown origin. While it stands a countdown ticks ' +
  'down every round — if it reaches zero you win the game instantly. But owning it ' +
  '<u>halves your maximum soldiers</u>, so guard it well: the enemy will come for it.';

export const STRANGE_DEVICE_SHOP_DESCRIPTION =
  'Starts a countdown — if it still stands at zero, you win! Only one can exist at a ' +
  'time. Warning: while built it <u>halves your soldier cap</u> (excess soldiers are ' +
  'disbanded at once). Build it on a defendable tile, not your headquarters.';
