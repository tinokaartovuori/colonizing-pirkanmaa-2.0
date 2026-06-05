/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: farm.h, header for Farm-class                                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef FARM_H
#define FARM_H

#include <memory>

#include "buildingbase.h"
#include "Core/resourcemaps.h"
#include "Tiles/grassland.h"


namespace Course {

/**
 * @brief The Farm class represents a farm building in the game.
 *
 * The player can grow crops on the farm which produces money. One worker is
 * needed to operate the farm. The farm has a growth cycle which is updated visually.
 * When growthPhase_ reaches 5 the crop is immediately harvested.
 */

class Farm : public BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Farm() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param owner points to the owning player.
     * @param buildcost is a ResourceMap of the building cost of the building
     * @param production is a ResourceMap of the production of the building
     *
     * @post Exception Guarantee: No guarantee.
     * @exception OwnerConflict - if the building conflicts with tile's
     * ownership.
     */
    explicit Farm(const std::weak_ptr<iGameEventHandler> &eventhandler,
            const std::weak_ptr<iObjectManager> &objectmanager,
            const std::weak_ptr<PlayerBase> &owner,
            const ResourceMap& buildcost = ConstResourceMaps::FARM_BUILD_COST,
            const ResourceMap& production = ConstResourceMaps::FARM_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~Farm() = default;


    /**
     * @brief Returns the building's type in string. In this case it's "Farm"
     * @return Building's type in string. In this case it's "Farm"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Gets the growth phase of the farm
     * @return Integer of farm's growth phase.
     * @post Exception guarantee: No-throw
     */
    int getGrowthPhase();


    /**
     * @brief Sets the growth phase of the farm. If the growth phase is set to
     *        five or more the phase resets to one.
     * @param Integer of farm's growth phase.
     * @post Exception guarantee: No-throw
     */
    void setGrowthPhase(int phase);


    /**
     * @brief If the tile has a farm, this function resets the crop
     *        growth cycle
     * @post Exception guarantee: ?????????????????????????????
     */
    void resetFarm();


private:
    int growthPhase_; //Value of the farm's growth phase


}; // class Farm

} // namespace Course


#endif // FARM_H
