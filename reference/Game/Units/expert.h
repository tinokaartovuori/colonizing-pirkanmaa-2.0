/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: expert.h, header for Expert-class                            #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef EXPERT_H
#define EXPERT_H

#include "unitbase.h"
#include "Core/resourcemaps.h"

namespace Student {

/**
 * @brief The Expert class represents an expert in the game.
 *
 * The expert can work in various buildings and is needed in nuclear and
 * hydroelectric power plants in order for them to operate. Adding the unit
 * costs money and the expert also has a salary that is payed when the round ends.
 * The unit can conquer new areas too if the tile has no owner
 */

class Expert : public Course::UnitBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    Expert() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     * @param tile is a shared pointer to the tile the unit is on
     */
    Expert(const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
             const std::weak_ptr<Course::iObjectManager> &objectmanager,
             const std::weak_ptr<GameSettingsManager> &gamesettingsmanager,
             const std::weak_ptr<Course::PlayerBase> &owner,
             const std::weak_ptr<Course::TileBase> &tile
             );


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     */
    Expert(const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
             const std::weak_ptr<Course::iObjectManager> &objectmanager,
             const std::weak_ptr<GameSettingsManager> &gamesettingsmanager,
             const std::weak_ptr<Course::PlayerBase> &owner);


    /**
     * @brief Default destructor.
     */
    virtual ~Expert() = default;


    /**
     * @brief Returns the unit's type in string. In this case it's "Expert"
     * @return Unit's type in string. In this case it's "Expert"
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


} //namespace Student



#endif // EXPERT_H
