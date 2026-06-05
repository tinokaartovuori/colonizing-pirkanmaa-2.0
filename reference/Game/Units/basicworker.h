/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: basicworker.h, header for BasicWorker-class                  #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef BASICWORKER_H
#define BASICWORKER_H

#include "unitbase.h"
#include "Core/resourcemaps.h"


namespace Course {

/**
 * @brief The BasicWorker class represents a "BasicWorker" in the game.
 *
 * The unit can work in various buildings and is needed to get resources
 * from the buildings. The worker also can cut down forest.
 * Adding the unit costs money and the worker also has a salary that is
 * payed when the round ends. The unit can conquer new areas
 * too if the tile has no owner
 */

class BasicWorker : public UnitBase
{
public:
    /**
     * @brief Disabled parameterless constructor.
     */
    BasicWorker() = delete;


    /**
     * @brief Constructor for the class.
     *
     * @param eventhandler points to the GameEventHandler.
     * @param objectmanager points to the ObjectManager.
     * @param gamesettingsmanager points to the GameSettingsManager.
     * @param owner is a shared pointer to the owner of the unit
     * @param tile is a shared pointer to the tile the unit is on
     */
    BasicWorker(const std::weak_ptr<iGameEventHandler>& eventhandler,
           const std::weak_ptr<iObjectManager>& objectmanager,
           const std::weak_ptr<Student::GameSettingsManager>& gamesettingsmanager,
           const std::weak_ptr<PlayerBase>& owner,
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
    BasicWorker(const std::weak_ptr<iGameEventHandler> &eventhandler,
           const std::weak_ptr<iObjectManager> &objectmanager,
           const std::weak_ptr<Student::GameSettingsManager> &gamesettingsmanager,
           const std::weak_ptr<PlayerBase> &owner
           );


    /**
     * @brief Default destructor.
     */
    virtual ~BasicWorker() = default;


    /**
     * @brief Returns the unit's type in string. In this case it's "BasicWorker"
     * @return Unit's type in string. In this case it's "BasicWorker"
     * @post Exception guarantee: No-throw
     */
    virtual std::string getType() const override;


    /**
     * @brief Returns the unit's salary
     * @return Unit's salary as a ResourceMap
     * @post Exception guarantee: ??????????????
     */
    virtual ResourceMap getSalary() override;


    /**
     * @brief Returns the cost of the unit
     * @return Unit's cost as a ResourceMap
     * @post Exception guarantee: ??????????????
     */
    virtual ResourceMap getCost() override;


}; // class BasicWorker

} // namespace Course


#endif // BASICWORKER_H
