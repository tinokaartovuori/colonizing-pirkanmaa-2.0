/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: outpost.h, header for Outpost-class                          #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef OUTPOST_H
#define OUTPOST_H

#include "buildingbase.h"
#include "Core/resourcemaps.h"


namespace Course {

/**
 * @brief The Outpost class represents an outpost in the game.
 *
 * It costs to build it and it consumes resources but it lets the player to
 * have more Soldiers
 *
 * Check Core/resourcemaps.h OUTPOST_PRODUCTION, OUTPOST_BUILD_COST
 * and OUTPOST_SOLDIER_VALUE for the effects.
 */

class Outpost : public BuildingBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Outpost() = delete;


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
    explicit Outpost(const std::weak_ptr<iGameEventHandler>& eventhandler,
            const std::weak_ptr<iObjectManager>& objectmanager,
            const std::weak_ptr<PlayerBase>& owner,
            const ResourceMap& buildcost = ConstResourceMaps::OUTPOST_BUILD_COST,
            const ResourceMap& production = ConstResourceMaps::OUTPOST_PRODUCTION
            );


    /**
     * @brief Default destructor.
     */
    virtual ~Outpost() = default;


    /**
     * @brief Returns the building's type in string.
     *        In this case it's "Outpost"
     * @return Building's type in string. In this case
     *         it's "Outpost"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;

    /**
     * @brief This function is used by the menu to get information about the
     *        headquarters building. The information is showed to the player
     *        as a text. This information tells us how much wood the tile has
     *        left and how many rounds the tile has been empty (cut).
     * @return String of the extra description the tile might have
     * @post Exception guarantee: ?????????????????????????????
     */
    std::string getExtraDescription();

}; // class Outpost

} // namespace Course


#endif // OUTPOST_H
