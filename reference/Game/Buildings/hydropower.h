/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: hydropower.h, header for HydroPower-class                    #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef HYDROPOWER_H
#define HYDROPOWER_H

#include "buildingbase.h"
#include "Core/resourcemaps.h"

namespace Student {

/**
 * @brief The HydroPower class represents a hydroelectric power plant in the game.
 *
 * It costs to build it and it consumes resources but produces money.
 * A basic worker is required to operate the plant.
 * An expert improves the efficiency.
 *
 * Check Core/resourcemaps.h HEPP_PRODUCTION and HEPP_BUILD_COST
 * for the specific resources
 */

class HydroPower : public Course::BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    HydroPower() = delete;


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
    explicit HydroPower(
            const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
            const std::weak_ptr<Course::iObjectManager>& objectmanager,
            const std::weak_ptr<Course::PlayerBase>& owner,
            const Course::ResourceMap& buildcost =
                  Course::ConstResourceMaps::HEPP_BUILD_COST,
            const Course::ResourceMap& production =
                  Course::ConstResourceMaps::HEPP_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~HydroPower() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Hydroelectric Power Plant"
     * @return Building's type in string. In this case
     *         it's "Hydroelectric Power Plant"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


}; // class HydroPower

} // namespace Student

#endif // HYDROPOWER_H
