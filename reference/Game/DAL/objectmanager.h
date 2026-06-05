/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: objectmanager.h, header to the ObjectManager-class           #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef OBJECTMANAGER_H
#define OBJECTMANAGER_H

#include <memory>
#include <vector>

#include "Core/playerbase.h"

#include "Interfaces/iobjectmanager.h"
#include "Interfaces/igameeventhandler.h"
#include "DAL/menuobjectmanager.h"
#include "Tiles/tilebase.h"

namespace Student {

#ifndef COURSE_OBJECTID
#define COURSE_OBJECTID
using ObjectId = unsigned int;
#endif

/**
 * @brief The ObjectManager class manages the tiles and various tile overlays
 */
class ObjectManager : public Course::iObjectManager
{
public:

    ObjectManager();


    /**
     * @brief Default destructor.
     */
    ~ObjectManager() = default;


    /**
     * @brief Sets pointer to the GameScene for the ObjectManager
     * @param Shared pointer to the game scene
     * @post Exception guarantee: No-throw
     */
    void setGameScene(std::shared_ptr<GameScene> gs) override;


    /**
     * @brief Returns pointer to the GameScene
     * @return Pointer to the GameScene
     * @post Exception guarantee: No-throw. Returns nullptr if scene is not set
     */
    std::shared_ptr<GameScene> getGameScene() override;


    /**
     * @brief Adds new tiles to the ObjectManager.
     * @param tiles is a vector that contains pointers to the Tiles to be added.
     * @post Exception guarantee: Basic
     */
    void addTiles(
            const std::vector<std::shared_ptr<Course::TileBase>>& tiles) override;


    /**
     * @brief Replaces a tile in the tiles_ vector with a new one
     * @param oldTile is a pointer to the tile to be removed.
     * @param newTile is a pointer to the tile to be added.
     * @post Exception guarantee: No guarantee
     */
    void replaceTile(std::shared_ptr<Course::TileBase> oldTile,
                    std::shared_ptr<Course::TileBase> newTile) override;


    /**
     * @brief Returns a shared pointer to a Tile that has specified coordinate.
     * @param coordinate Requested Tile's Coordinate
     * @return a pointer to a Tile that has the given coordinate.
     * If no for the coordinate exists, return empty pointer.
     * @post Exception Guarantee: Basic
     */
    std::shared_ptr<Course::TileBase> getTile(
            const Course::Coordinate& coordinate) override;


    /**
     * @brief Returns a vector of shared pointers to Tiles specified by
     * a vector of Coordinates.
     * @param coordinates a vector of Coordinates for the requested Tiles
     * @return Vector of that contains pointers to Tile's that match
     * the coordinates. The vector is empty if no matches were made.
     * @post Exception Guarantee: Basic
     */
    std::vector<std::shared_ptr<Course::TileBase>> getTiles() override;


    /**
     * @brief Sets a pointer to an object that is on the tile the mouse pointer
     *        is on
     * @param coordinates a vector of Coordinates for the requested Tiles
     * @return Vector of that contains pointers to Tile's that match
     *         the coordinates. The vector is empty if no matches were made.
     * @post Exception Guarantee: No-throw
     */
    void setHoverBorder(const
                      std::shared_ptr<Student::MouseHoverBorder> border) override;

    /**
     * @brief Sets a pointer to an object that shows the tile that was clicked
     * @param Pointer to the tile the border is wanted to be drawn
     * @post Exception Guarantee: No guarantee
     */
    void setClickedTileBorder(std::shared_ptr<Course::TileBase> tile) override;

    /**
     * @brief Gets the pointer to an object that shows the tile that was clicked
     * @return Pointer to the object that shows the tile that was clicked
     * @post Exception Guarantee: No-throw
     */
    std::shared_ptr<Student::ClickedTileBorder> getClickedTileBorder() override;

    /**
     * @brief Removes the pointer to the object that shows the tile that was clicked
     * @post Exception Guarantee: No-throw
     */
    void removeClickedTileBorder() override;

    /**
     * @brief Returns the pointer to the object that shows the tile the mouse
     *        pointer is on
     * @return Pointer to the border that is below the mouse cursor
     * @post Exception Guarantee: No-throw
     */
    std::shared_ptr<Student::MouseHoverBorder> getBorderTile() override;


    /**
     * @brief Adds the pointers to the data access layers to the object manager
     * @post Exception Guarantee: ??????????
     */
    void addDALS(const std::shared_ptr<Course::iGameEventHandler> gameeventhandler,
         const std::shared_ptr<Student::iMenuObjectManager> menuobjectmanager,
         const std::shared_ptr<Student::GameSettingsManager> gamesettingsmanager)
         override;


    /**
     * @brief Returns a vector of pointers to the tiles that are connected
     *        to the given player's headquarters
     * @param Pointer to a player object
     * @return A vector of pointers to the tiles that are connected
     *         to the given player's headquarters
     * @post Exception Guarantee: Strong
     */
    std::vector<std::shared_ptr<Course::TileBase>>
         getHqConnectedTiles(std::shared_ptr<Course::PlayerBase> player) override;


    /**
     * @brief Returns a pointer to the tile that has the given player's
     *        headquarters
     * @param Pointer to a player object
     * @return A pointer to the tile that has the given player's
     *         headquarters
     * @post Exception Guarantee: Strong
     */
    std::shared_ptr<Course::TileBase>
                  getHqTile(std::shared_ptr<Course::PlayerBase> player) override;


    /**
     * @brief Returns a vector of pointers to the tiles the current player
     *        can place an unit
     * @return A vector of pointers to the tiles the current player
     *         can place an unit
     * @post Exception Guarantee: No guarantee
     */
    std::vector<std::shared_ptr<Course::TileBase> > getAvailableTiles() override;


    /**
     * @brief Adds the overlay on the gamescene that shows the tiles the
     *        current player cannot place units on
     * @post Exception Guarantee: No guarantee
     */
    void addBlockTileOverlays() override;


    /**
     * @brief Removes the overlay on the gamescene that shows the tiles the
     *        current player cannot place units on
     * @post Exception Guarantee: No guarantee
     */
    void removeBlockTileOverlays() override;


    /**
     * @brief Returns the number of tiles the objectmanager (and scene) has
     * @return Integer of the number of tiles the scene has
     * @post Exception Guarantee: No-throw
     */
    int getTileCount() override;


    /**
     * @brief Returns the number of tiles a specific player has in
     *        the objectmanager (and scene)
     * @param Pointer to a player object
     * @return Integer of the number of tiles a specific player has in
     *         the objectmanager (and scene)
     * @post Exception Guarantee: No guarantee
     */
    int getTileCountForPlayer(std::shared_ptr<Course::PlayerBase> player) override;


    /**
     * @brief Returns the number of tiles the objectmanager (and scene) has that
     *        are owned by no one
     * @return Integer of the number of neutral tiles the scene has
     * @post Exception Guarantee: No guarantee
     */
    int getNeutralTiles() override;

private:

    std::vector<std::shared_ptr<Course::TileBase>> tiles_;

    //Pointer to the border that shows the tile the mouse pointer is on
    std::shared_ptr<Student::MouseHoverBorder> hoverBorder_;

    //Pointer to the border that shows the tile that has been clicked
    std::shared_ptr<Student::ClickedTileBorder> clickedTileBorder_;

    /*Vector of the pointers to the overlays that are on top of the tiles
     *on which the player cannot place an unit */
    std::vector<std::shared_ptr<Student::BlockedTile>> blockedTileOverlays_;

    std::weak_ptr<Course::iGameEventHandler> gameEventHandler_;
    std::weak_ptr<Student::iMenuObjectManager> menuObjectManager_;
    std::weak_ptr<Student::GameSettingsManager> gameSettingsManager_;
    std::weak_ptr<GameScene> gameScene_;



}; // class ObjectManager

} // namespace Student


#endif // ObjectManager_H
