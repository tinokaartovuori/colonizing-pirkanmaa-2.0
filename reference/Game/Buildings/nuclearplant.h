/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: nuclearplant.h, header for NuclearPlant-class                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef NUCLEARPLANT_H
#define NUCLEARPLANT_H

#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Student {

/**
 * @brief The NuclearPlant class represents a nuclear power plant in the game.
 *
 * It costs to build it and it consumes resources but produces lots of gold.
 * An expert and a basic worker is required to operate the plant. A second
 * worker doubles the production
 *
 * Check Core/resourcemaps.h NUCLEARPP_PRODUCTION and NUCLEARPP_BUILD_COST
 * for the specific resources
 */

class NuclearPlant : public Course::BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    NuclearPlant() = delete;


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
    explicit NuclearPlant(
            const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
            const std::weak_ptr<Course::iObjectManager>& objectmanager,
            const std::weak_ptr<Course::PlayerBase>& owner,
            const Course::ResourceMap& buildcost =
                    Course::ConstResourceMaps::NUCLEARPP_BUILD_COST,
            const Course::ResourceMap& production =
                    Course::ConstResourceMaps::NUCLEARPP_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~NuclearPlant() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Nuclear Power Plant"
     * @return Building's type in string. In this case
     *         it's "Nuclear Power Plant"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


}; // class NuclearPlant

} // namespace Student


#endif // NUCLEARPLANT_H
