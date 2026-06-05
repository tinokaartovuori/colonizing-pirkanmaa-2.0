/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gameeventhandler.h, header to the GameEventHandler-class     #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef GAMEEVENTHANDLER_H
#define GAMEEVENTHANDLER_H

#include <memory>
#include <vector>

#include "Interfaces/igameeventhandler.h"
#include "DAL/playermanager.h"
#include "DAL/menuobjectmanager.h"
#include "DAL/gamesettingsmanager.h"

#include "Core/resourcemaps.h"

#include "Graphics/gamescene.h"
#include "Graphics/sceneitem.h"

#include "Buildings/bridge.h"
#include "Buildings/buildingbase.h"
#include "Buildings/farm.h"
#include "Buildings/headquarters.h"
#include "Buildings/hydropower.h"
#include "Buildings/mine.h"
#include "Buildings/nuclearplant.h"
#include "Buildings/outpost.h"
#include "Buildings/village.h"

#include "Units/basicworker.h"
#include "Units/expert.h"
#include "Units/soldier.h"

namespace Student {

/**
 * @brief The GameEventHandler class manages various events in the game
 *
 */
class GameEventHandler : public Course::iGameEventHandler,
                         public std::enable_shared_from_this<GameEventHandler>
{

public:

    GameEventHandler();


    /**
     * @brief Constructor for the class.
     * @param objectmanager points to the ObjectManager
     * @param playermanager points to the PlayerManager
     * @param menuobjectmanager points to the MenuObjectManager
     * @param gamesettingsmanager points to the GameSettingsManager
     * @post Exception guarantee: No guarantee.
     */

    GameEventHandler(std::shared_ptr<Student::ObjectManager> objectmanager,
              std::shared_ptr<Student::PlayerManager> playermanager,
              std::shared_ptr<Student::MenuObjectManager> menuobjectmanager,
              std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager);


    /**
     * @brief Default destructor.
     */
    ~GameEventHandler() = default;


    /**
     * @brief Sets a pointer to the gamescene and saves it into the event handler
     * @param Pointer to the gamescene
     * @post Exception guarantee: Strong
     */
    void setGameScene(std::shared_ptr<GameScene> gs) override;


    /**
     * @brief Handles the actions done in the first round. This includes placing
     *        the headquarters and getting the nine closest tiles if no one
     *        owns them
     * @param Pointer to the tile the headquarters is added on
     * @post Exception guarantee: No guarantee
     */
    void firstRoundActions(std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Handles the actions done when a tile is clicked
     * @param Pointer to the tile that was clicked
     * @post Exception guarantee: No guarantee
     */
    void tileClicked(std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Handles the actions done when turn is ended. This includes
     *        generating resources, paying salaries for units, conquering
     *        tiles, checking if there's tiles that are not connected to
     *        the headquarters and checking if someone lost.
     * @post Exception guarantee: No guarantee
     */
    void endTurn() override;


    /**
     * @brief Neutralizes the player by removing the ownership of players' tiles,
     *        destroying the headquarters and resetting the player's farm
     * @param Pointer to the player who is wanted to be nautralized
     * @post Exception guarantee: No guarantee
     */
    void neutralizePlayer(std::shared_ptr<Course::PlayerBase> player) override;


    /**
     * @brief Changes animated tile to be stuck in a specific frame
     * @param tile points to the tile that is wanted to be set static
     * @param frame is an integer of the frame that is wanted to be
     *        set for the tile
     * @post Exception guarantee: No guarantee
     */
    void updateAnimatedTileToStatic(std::shared_ptr<Course::TileBase> tile,
                                                        int frame) override;


    /**
     * @brief Updates the forest visually to be either grown or cut.
     *        The function can also change the tile type into grassland
     *        and set a building on it.
     * @param status is a string that tells the function what to do.
     *        "Cut" updates the tile to be visually cut, "Grow" updates the tile
     *  to be visually
     *        grown and "Grassland" converts the tile into grassland
     * @param tile points to the tile that is wanted to be changed
     * @param building points to a building is wanted to be built on the tile.
     *        This is only possible after the tile is converted into grassland
     * @post Exception guarantee: No guarantee
     */
    void updateForest(std::string status,
               std::shared_ptr<Course::TileBase> tile,
               const std::shared_ptr<Course::BuildingBase>& building = nullptr)
               override;


    /**
     * @brief Sets up the menu view that shows information about a tile that
     *        is clicked and the buildings that can be built on the tile
     * @param tile points to the tile that is clicked
     * @param index_for_buildings is an index for the building that is
     *        visible in the menu
     * @post Exception guarantee: No guarantee
     */
    void setTileInspectionMenuView(std::shared_ptr<Course::TileBase> tile,
                                   int index_for_buildings = 0) override;


    /**
     * @brief Opens the menu view that shows how many rounds have been played and
     *        how much of map area players own
     * @post Exception guarantee: No guarantee
     */
    void openStatsMenuView() override;


    /**
     * @brief Opens the menu view that shows how many rounds have been played and
     *        how much of map area players own
     * @post Exception guarantee: No guarantee
     */
    void openDefaultMenuView() override;


    /**
     * @brief Opens the unit buy menu
     * @post Exception guarantee: No guarantee
     */
    void openUnitBuyMenu() override;


    /**
     * @brief Creates the shared pointer for the unit to be created
     * @param string of the unit type to be created
     * @post Exception guarantee: No guarantee
     */
    void createUnit(std::string unit) override;


    /**
     * @brief Is called when the "move" button is clicked on the menu for
     *        an unit. The function keeps track of the tile the unit was picked
     *        from and which unit it was.
     * @param index is the index of the unit either in the units_ or
     *        conqueringUnits_ vector
     * @param tile points to the tile the unit is picked from
     * @post Exception guarantee: No guarantee
     */
    void moveUnitFromTile(int index,
                        std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Removes resources from the current player according to the
     *        purchase cost of the given object.
     * @param pointer to the object to be bought
     * @post Exception guarantee: No guarantee
     */
    void buyUnitOrBuilding(
            std::shared_ptr<Course::PlaceableGameObject> object) override;


    /**
     * @brief Checks if the current player has enough resources to buy
     *        the given object
     * @param pointer to the object to be bought
     * @post Exception guarantee: No guarantee
     */
    bool canBuyUnitOrBuilding(
            std::shared_ptr<Course::PlaceableGameObject> object) override;


    /**
     * @brief Deletes unit from tile and excecutes the required actions
     * @param index is the index of the unit either in the units_ or
     *        conqueringUnits_ vector to be removed
     * @param tile points to the tile the unit is removed from
     * @post Exception guarantee: No guarantee
     */
    void deleteUnitFromTile
                (int index, std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Deletes unit from tile and excecutes the required actions
     * @param unit points to the unit to be removed
     * @param tile points to the tile the unit is removed from
     * @post Exception guarantee: No guarantee
     */
    void deleteUnitFromTile(std::shared_ptr<Course::UnitBase> unit,
                            std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Calls gamescene to update the given tile graphically
     * @param tile points to the tile to be updated
     * @post Exception guarantee: No guarantee
     */
    void updateTile(std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Creates a shared pointer to the building that is wanted to
     *        be built and removes the purchase cost from the current
     *        player if the player can addord the building,
     * @param building_string is the building type in string
     * @param tile points to the tile that the building is wanted to added
     * @post Exception guarantee: No guarantee
     */
    void buildBuilding(std::string building_string,
                                std::shared_ptr<Course::TileBase> tile) override;


    /**
     * @brief Returns a Resource map of the current revenue the current player
     *        has when the turn ends. This function is used by the menu
     * @return resource map of the current revenue the current player has
     * @post Exception guarantee: No guarantee
     */
    Course::ResourceMap getCurrentRevenue() override;


    /**
     * @brief Returns a Resource map of the current expenses the current player
     *        has when the turn ends. This function is used by the menu
     * @return resource map of the current expenses the current player has
     * @post Exception guarantee: No guarantee
     */
    Course::ResourceMap getCurrentExpences() override;


    /**
     * @brief Returns a Resource map of the current net revenue the current player
     *        will earn when the turn ends. In other words it's revenue summed
     *        with expenses. This function is used by the menu.
     * @return resource map of the current revenue the current player has
     * @post Exception guarantee: No guarantee
     */
    Course::ResourceMap getCurrentNet() override;


    /**
     * @brief Returns a pointer to the player obejct currently in turn
     * @return pointer to the player obejct currently in turn
     * @post Exception guarantee: No guarantee
     */
    std::shared_ptr<Course::PlayerBase> getCurrentPlayer() override;


    /**
     * @brief Calls mainwindow to restart the game
     * @post Exception guarantee: No guarantee
     */
    void restartGame() override;


private:

    std::weak_ptr<Student::ObjectManager> objectManager_;
    std::weak_ptr<Student::PlayerManager> playerManager_;
    std::weak_ptr<Student::MenuObjectManager> menuObjectManager_;
    std::weak_ptr<Student::GameSettingsManager> gameSettingsManager_;
    std::weak_ptr<GameScene> gameScene_;

    //Unit to be deployed
    std::shared_ptr<Course::UnitBase> unitToDeploy_;

    //The tile the unit is moved from
    std::shared_ptr<Course::TileBase> unitPreviousTile_;

}; // class GameEventHandler

} // namespace Course


#endif // GameEventHandler_H
