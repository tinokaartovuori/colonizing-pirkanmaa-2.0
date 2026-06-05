/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: igameeventhandler.h, interface for GameEventHandler          #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef IGAMEEVENTHANDLER_H
#define IGAMEEVENTHANDLER_H

#include <memory>
#include <vector>

#include "Core/basicresources.h"
#include "Core/playerbase.h"

namespace Student {
class PlayerManager;
class GameScene;
}
namespace Course {

class TileBase;
class UnitBase;
class BuildingBase;

class PlaceableGameObject;

/**
 * @brief The iGameEventHandler class is an interface which the Course-side
 * code uses to interact with the GameEventHandler implemented by the students.
 *
 * @note The interface declares only functions required by the Course-side code.
 * The actual implementation can (and should!) contain more stuff.
 * @note In a "real" project, the GameEventHandler should be a singleton
 * and not use an abstract base class to define the interface for it.
 * <b>This design was chosen merely for pedagogical reasons and to
 * give students more freedom in their project design.</b>
 */
class iGameEventHandler : public QObject
{
    Q_OBJECT

signals:
    void restartGameSignal();


public:

    /**
     * @brief Default destructor.
     */
    virtual ~iGameEventHandler() = default;

    virtual void setGameScene(std::shared_ptr<Student::GameScene> gs) = 0;

    virtual void firstRoundActions(std::shared_ptr<Course::TileBase> tile) = 0;

    virtual void tileClicked(std::shared_ptr<Course::TileBase> tile) = 0;

    virtual void endTurn() = 0;

    virtual void updateAnimatedTileToStatic(std::shared_ptr<Course::TileBase> tile,
                            int frame) = 0;

    virtual void updateForest(std::string status,
              std::shared_ptr<Course::TileBase> tile,
              const std::shared_ptr<Course::BuildingBase>& building = nullptr) = 0;


    virtual void setTileInspectionMenuView(std::shared_ptr<Course::TileBase> tile,
                                           int index_for_buildings = 0) = 0;

    virtual void openStatsMenuView() = 0;

    virtual void openDefaultMenuView() = 0;

    virtual void createUnit(std::string unit) = 0;

    virtual void openUnitBuyMenu() = 0;

    virtual void moveUnitFromTile(int index,
                                std::shared_ptr<Course::TileBase> tile) = 0;

    virtual void deleteUnitFromTile(int index,
                                    std::shared_ptr<Course::TileBase> tile) = 0;

    virtual void deleteUnitFromTile(std::shared_ptr<Course::UnitBase> unit,
                            std::shared_ptr<Course::TileBase> tile) = 0;

    virtual void updateTile(std::shared_ptr<Course::TileBase> tile) = 0;

    virtual void buyUnitOrBuilding(
            std::shared_ptr<Course::PlaceableGameObject> object) = 0;

    virtual bool canBuyUnitOrBuilding(
            std::shared_ptr<Course::PlaceableGameObject> object) = 0;

    virtual void buildBuilding(std::string building,
                               std::shared_ptr<Course::TileBase> tile) = 0;

    virtual Course::ResourceMap getCurrentRevenue() = 0;

    virtual Course::ResourceMap getCurrentExpences() = 0;

    virtual Course::ResourceMap getCurrentNet() = 0;

    virtual std::shared_ptr<Course::PlayerBase> getCurrentPlayer() = 0;

    virtual void neutralizePlayer(std::shared_ptr<Course::PlayerBase> player) = 0;

    virtual void restartGame() = 0;


}; // class iGameEventHandler

} // namespace Course


#endif // IGAMEEVENTHANDLER_H
