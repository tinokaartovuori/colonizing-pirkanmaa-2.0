/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: mine.h, header for Mine-class                                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MINE_H
#define MINE_H


#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Student {

/**
 * @brief The Mine class represents a mine in the game.
 *
 * It costs to build it and it consumes resources but produces money,
 * stone and metal. A basic worker is required to operate the mine.
 * An expert improves the efficiency.
 *
 * Check Core/resourcemaps.h MINE_PRODUCTION and MINE_BUILD_COST
 * for the specific resources
 */

class Mine : public Course::BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Mine() = delete;


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
    explicit Mine(const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
            const std::weak_ptr<Course::iObjectManager> &objectmanager,
            const std::weak_ptr<Course::PlayerBase> &owner,
            const Course::ResourceMap& buildcost =
                    Course::ConstResourceMaps::MINE_BUILD_COST,
            const Course::ResourceMap& production =
                    Course::ConstResourceMaps::MINE_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~Mine() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Mine"
     * @return Building's type in string. In this case it's "Mine"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


}; // class Mine

} // namespace Student

#endif // MINE_H
