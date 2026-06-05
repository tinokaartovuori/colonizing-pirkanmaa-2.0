/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: soldier.h, header for Soldier-class                          #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef SOLDIER_H
#define SOLDIER_H

#include "unitbase.h"
#include "Core/resourcemaps.h"

namespace Student {

/**
 * @brief The Soldier class represents a soldier in the game.
 *
 * The soldier can conquer tiles owned by another player if the tile has more 
 * soldiers than the opponent. Adding the unit costs money and metal.
 * A salary has to be payed when the round ends.
 */

class Soldier : public Course::UnitBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Soldier() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     * @param tile is a shared pointer to the tile the unit is on
     */
    Soldier(const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
            const std::weak_ptr<Course::iObjectManager>& objectmanager,
            const std::weak_ptr<GameSettingsManager>& gamesettingsmanager,
            const std::weak_ptr<Course::PlayerBase>& owner,
            const std::weak_ptr<Course::TileBase>& tile
            );


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     */
    Soldier(const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
            const std::weak_ptr<Course::iObjectManager> &objectmanager,
            const std::weak_ptr<GameSettingsManager> &gamesettingsmanager,
            const std::weak_ptr<Course::PlayerBase> &owner);


    /**
     * @brief Default destructor.
     */
    virtual ~Soldier() = default;


    /**
     * @brief Returns the unit's type in string. In this case it's "Soldier"
     * @return Unit's type in string. In this case it's "Soldier"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Returns the unit's salary
     * @return Unit's salary as a ResourceMap
     * @post Exception guarantee: ??????????????
     */
    virtual Course::ResourceMap getSalary() override;


    /**
     * @brief Returns the cost of the unit
     * @return Unit's cost as a ResourceMap
     * @post Exception guarantee: ??????????????
     */
    virtual Course::ResourceMap getCost() override;

}; // class Expert



} // namespace Soldier



#endif // SOLDIER_H
