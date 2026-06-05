/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: mikontalo.h, header for Mikontalo-class                      #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MIKONTALO_H
#define MIKONTALO_H


#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Student {

/**
 * @brief The Mikontalo class represents a Mikontalo in the game.
 *
 * It costs to build it and it consumes resources but it lets the player to
 * have more BasicWorkers and Experts.
 *
 * Check Core/resourcemaps.h MIKONTALO_PRODUCTION, MIKONTALO_BUILD_COST
 * and Mikontalo_UNIT_VALUE for the effects.
 */

class Mikontalo : public Course::BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Mikontalo() = delete;


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
    explicit Mikontalo(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
            const std::weak_ptr<Course::iObjectManager>& objectmanager,
            const std::shared_ptr<Course::PlayerBase> &owner,
            const Course::ResourceMap& buildcost =
                    Course::ConstResourceMaps::NO_RESOURCES,
            const Course::ResourceMap& production =
                    Course::ConstResourceMaps::NO_RESOURCES
            );


    /**
     * @brief Default destructor.
     */
    virtual ~Mikontalo() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Mikontalo"
     * @return Building's type in string. In this case
     *         it's "Mikontalo"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;

    /**
     * @brief Returns the building's extra description in string.
     * @return Building's extra description in string.
     * @post Exception guarantee: No-throw
     */
    std::string getExtraDescription();


}; // class Mikontalo

} // namespace Student

#endif // MIKONTALO_H
