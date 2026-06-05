/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: iobjectmanager.h, interface for ObjectManager                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef IOBJECTMANAGER_H
#define IOBJECTMANAGER_H

#include <memory>
#include <vector>

#include "Core/basicresources.h"
#include "Core/coordinate.h"


#include "Menus/menuview.h"
#include "Menus/button.h"
#include "Menus/menuobjectcontainer.h"
#include "Overlays/mousehoverborder.h"
#include "Overlays/clickedtileborder.h"
#include "Overlays/blockedtile.h"
#include "Graphics/imagevectors.h"
#include "Graphics/animationoptions.h"

namespace Student {
    class iMenuObjectManager;
    class GameSettingsManager;
    class GameScene;

#ifndef COURSE_OBJECTID
#define COURSE_OBJECTID
using ObjectId = unsigned int;
#endif

}

namespace Course {


#ifndef COURSE_OBJECTID
#define COURSE_OBJECTID
using ObjectId = unsigned int;
#endif

/**
 * @brief The iObjectManager class is an interface which the Course-side
 * code uses to interact with the ObjectManager implemented by the students.
 *
 * @note The interface declares only functions required by the Course-side code.
 * The actual implementation can (and should!) contain more stuff.
 */
class iObjectManager : public std::enable_shared_from_this<iObjectManager>
{
public:

    /**
     * @brief Default destructor.
     *
     */

    virtual ~iObjectManager() = default;

    /**
     * @brief Adds new tiles to the ObjectManager.
     * @param tiles a vector that contains the Tiles to be added.
     * @post The tile-pointers in the vector are stored in the ObjectManager.
     * Exception Guarantee: Basic
     *
     */
    virtual void addTiles(
            const std::vector<std::shared_ptr<TileBase>>& tiles) = 0;

    virtual void replaceTile(std::shared_ptr<Course::TileBase> oldTile,
                            std::shared_ptr<Course::TileBase> newTile) = 0;

    /**
     * @brief Returns a shared pointer to a Tile that has specified coordinate.
     * @param coordinate Requested Tile's Coordinate
     * @return a pointer to a Tile that has the given coordinate.
     * If no for the coordinate exists, return empty pointer.
     * @post Exception Guarantee: Basic
     */
    virtual std::shared_ptr<TileBase> getTile(
            const Coordinate& coordinate) = 0;


    /**
     * @brief Returns a vector of shared pointers to Tiles specified by
     * a vector of Coordinates.
     * @param coordinates a vector of Coordinates for the requested Tiles
     * @return Vector of that contains pointers to Tile's that match
     * the coordinates. The vector is empty if no matches were made.
     * @post Exception Guarantee: Basic
     */

    //// OWN IMPLEMENTATIONS
    ///
    virtual std::vector<std::shared_ptr<TileBase>> getTiles() = 0;

    virtual void setHoverBorder(const
                      std::shared_ptr<Student::MouseHoverBorder> border) = 0;

    virtual void setClickedTileBorder(std::shared_ptr<Course::TileBase> tile) = 0;

    virtual std::shared_ptr<Student::ClickedTileBorder> getClickedTileBorder() = 0;

    virtual void removeClickedTileBorder() = 0;

    virtual std::shared_ptr<Student::MouseHoverBorder> getBorderTile() = 0;

    virtual void addDALS(const std::shared_ptr
                         <Course::iGameEventHandler> gameeventhandler,
                 const std::shared_ptr
                         <Student::iMenuObjectManager> menuobjectmanager,
                 const std::shared_ptr
                         <Student::GameSettingsManager> gamesettingsmanager) = 0;

    virtual void setGameScene(std::shared_ptr<Student::GameScene> gs) = 0;

    virtual std::vector<std::shared_ptr<Course::TileBase>>
               getHqConnectedTiles(std::shared_ptr<Course::PlayerBase> player) = 0;

    virtual std::shared_ptr<Course::TileBase>
                          getHqTile(std::shared_ptr<Course::PlayerBase> player) = 0;

    virtual std::vector<std::shared_ptr<Course::TileBase> > getAvailableTiles() = 0;

    virtual void addBlockTileOverlays() = 0;

    virtual void removeBlockTileOverlays() = 0;

    virtual std::shared_ptr<Student::GameScene> getGameScene() = 0;

    virtual int getTileCount() = 0;

    virtual int getTileCountForPlayer(std::shared_ptr<Course::PlayerBase> player) = 0;

    virtual int getNeutralTiles() = 0;

}; // class iObjectManager

} // namespace Course
#endif // IOBJECTMANAGER_H
