/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: bridge.h, header for Bridge-class                            #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef BRIDGE_H
#define BRIDGE_H


#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Student {

/**
 * @brief The Bridge class represents a bridge in the game.
 *
 * The bridge is needed to conquer new areas if a river is on the player's border
 */

class Bridge : public Course::BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Bridge() = delete;


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
    explicit Bridge(
            const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
            const std::weak_ptr<Course::iObjectManager>& objectmanager,
            const std::weak_ptr<Course::PlayerBase>& owner,
            const Course::ResourceMap& buildcost =
                    Course::ConstResourceMaps::BRIDGE_BUILD_COST,
            const Course::ResourceMap& production =
                    Course::ConstResourceMaps::BRIDGE_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~Bridge() = default;


    /**
     * @brief Returns the building's type in string. In this case it's "Bridge"
     * @return Building's type in string. In this case it's "Bridge"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;

}; // class Bridge

} // namespace Student

#endif // BRIDGE_H
