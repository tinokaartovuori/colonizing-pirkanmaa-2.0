/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: village.h, header for Village-class                          #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef NEIGHBOURHOOD_H
#define NEIGHBOURHOOD_H


#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Student {

/**
 * @brief The Village class represents a village in the game.
 *
 * It costs to build it and it consumes resources but it lets the player to
 * have more BasicWorkers and Experts.
 *
 * Check Core/resourcemaps.h VILLAGE_PRODUCTION, VILLAGE_BUILD_COST
 * and VILLAGE_UNIT_VALUE for the effects.
 */

class Village : public Course::BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Village() = delete;


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
    explicit Village(
            const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
            const std::weak_ptr<Course::iObjectManager>& objectmanager,
            const std::weak_ptr<Course::PlayerBase>& owner,
            const Course::ResourceMap& buildcost =
                    Course::ConstResourceMaps::VILLAGE_BUILD_COST,
            const Course::ResourceMap& production =
                    Course::ConstResourceMaps::VILLAGE_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~Village() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Village"
     * @return Building's type in string. In this case
     *         it's "Village"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;

    /**
     * @brief Returns the building's extra description in string.
     * @return Building's extra description in string.
     * @post Exception guarantee: No-throw
     */
    std::string getExtraDescription();


}; // class Village

} // namespace Student

#endif // NEIGHBOURHOOD_H
