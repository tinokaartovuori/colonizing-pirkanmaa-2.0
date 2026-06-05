/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gamescene.h, header for GameScene-class                      #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef SIMPLEGAMESCENE_H
#define SIMPLEGAMESCENE_H

#include <QGraphicsScene>
#include <QGraphicsView>
#include <QGraphicsPixmapItem>
#include <QMouseEvent>
#include <QEvent>
#include <QTimer>
#include <QImage>
#include <QDebug>

#include <map>
#include <memory>

#include "Core/baseobject.h"
#include "Core/menuobject.h"
#include "Core/gameobject.h"

#include "Menus/menuview.h"
#include "Menus/menuobjectcontainer.h"

#include "freesceneitem.h"

#include "Overlays/mousehoverborder.h"
#include "Overlays/clickedtileborder.h"

#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Interfaces/imenuobjectmanager.h"

#include "DAL/gamesettingsmanager.h"



namespace Student {

class SceneItem;

/**
 * @brief The GameScene class manages graphical items on the gamescene and draws
 *        them with the help of QGraphicsScene
 */
class GameScene : public QGraphicsScene
{
    Q_OBJECT

signals:
    /**
     * @brief Calls mainwidow to refresh the scene. This is done
     *        by emitting the signal
     */
    void updateScene();

public:
    /**
     * @brief Constructor for the class.
     *
     * @param qt_parent points to the parent object per Qt's parent-child-system.
     * @param width in tiles for the game map.
     * @param height in tiles for the game map.
     * @param scale is the size in pixels of a single square tile.
     *
     * @pre 0 < width <= 100 && 0 < height <= 100 && 0 < scale <= 500. Otherwise
     * default values are used for the created object.
     */
    GameScene(QWidget* qt_parent,
              std::shared_ptr<Course::iGameEventHandler> eventHandler,
              std::shared_ptr<Course::iObjectManager> objectManager,
              std::shared_ptr<Student::iMenuObjectManager> menuObjectManager,
              std::shared_ptr<Student::GameSettingsManager> gameSettingsManager);

    /**
     * @brief Destructor.
     */
    ~GameScene() = default;


    /**
     * @brief Deletes the shared pointer references of SceneItems and
     *        QTimer object
     * @post Exception guarantee: No guarantee
     */
    void deleteObjects();


    /**
     * @brief Adds a scene item into the sceneItems_ vector
     * @param pointer to the sceneItem to be added
     * @post Exception guarantee: Strong
     */
    void addSceneItem(std::shared_ptr<SceneItem> sceneItem);


    /**
     * @brief Draws tiles and buildings on the gameScene. The function
     *        creates a mapSceneItem and passes various variables to it
     * @param pointer to the GameObject to be drawn
     * @post Exception guarantee: No guarantee
     */
    void drawItem(std::shared_ptr<Course::GameObject> obj);


    /**
     * @brief Draws units on the gameScene. The function
     *        creates an unitSceneItem and passes various variables to it
     * @param pointer to the UnitBase to be drawn
     * @post Exception guarantee: No guarantee
     */
    void drawItem(std::shared_ptr<Course::UnitBase> obj);


    /**
     * @brief Draws MenuView on the gameScene. The MenuView is the main
     *        background of the menu and other menu graphics are drawn on top
     *        of it
     * @param pointer to the MenuView to be drawn
     * @post Exception guarantee: No guarantee
     */
    void drawItem(std::shared_ptr<MenuView> obj);


    /**
     * @brief Draws MenuObjectContainer on the menu
     * @param obj points to the MenuObjectContainer to be drawn
     * @param cont points to the container interface which makes
     *        possible to insert containers inside another container.
     *        It also makes relative coordinates possible
     * @post Exception guarantee: No guarantee
     */
    void drawItem(std::shared_ptr<MenuObjectContainer> obj,
                  std::shared_ptr<iContainer> cont);


    /**
     * @brief Draws MenuObjects such as buttons and labels
     * @param obj points to the MenuObject to be drawn
     * @param cont points to the container interface which makes
     *        possible to insert containers inside another container.
     *        It also makes relative coordinates possible
     * @post Exception guarantee: No guarantee
     */
    void drawItem(std::shared_ptr<MenuObject> obj,
                  std::shared_ptr<iContainer> cont);


    /**
     * @brief Draws items that follow the mouse cursor. It could be implemented
     *        to any item but is used for units at the moment
     * @param event points to the event that happened on the scene
     * @post Exception guarantee: No guarantee
     */
    void drawMouseFollowItem(QEvent *event);


    /**
     * @brief Updates the game object's location and images if they've changed
     * @param pointer to the GameObject to be updated graphically
     * @post Exception guarantee: No guarantee
     */
    void updateItem(std::shared_ptr<Course::GameObject> obj);


    /**
     * @brief Updates menuobject's location
     * @param pointer to the MenuObject to be updated graphically
     * @post Exception guarantee: No guarantee
     */
    void updateItem(std::shared_ptr<MenuObject> obj);


    /**
     * @brief Updates a tile in the game graphically by drawing a building
     *        on it if a building is built. The function can also draw
     *        and remove units as is required
     * @param pointer to the tile to be updated
     * @post Exception guarantee: No guarantee
     */
    void updateTile(std::shared_ptr<Course::TileBase> tileobj);


    /**
     * @brief Removes a GameObect from the game scene
     * @param pointer to the GameObect to be removed
     * @post Exception guarantee: No guarantee
     */
    void removeItem(std::shared_ptr<Course::BaseObject> obj);


    /**
     * @brief Removes a menu container from the game scene
     * @param pointer to the iContainer to be removed
     * @post Exception guarantee: No guarantee
     */
    void removeContainer(std::shared_ptr<iContainer> cont);


    /**
     * @brief Removes the item that follows the mouse cursor
     * @post Exception guarantee: No-throw
     */
    void removeMouseFollowItem();


    /**
     * @brief Removes a menu container from the game scene
     * @param vector of the filepaths to the item's graphics (multiple frames)
     * @post Exception guarantee: Strong
     */
    void addMouseFollowPicture(std::vector<std::string> imagevector);


    /**
     * @brief Checks if the given item is on the scene
     * @param pointer to the object
     * @return True: the object is in the scene
     *         False: the item is not in the scene
     * @post Exception guarantee: No guarantee
     */
    bool isObjectInScene(std::shared_ptr<Course::BaseObject> obj);


    /**
     * @brief Returns a pointer to an object in the scene
     * @param pointer to the BaseObject
     * @return pointer to a object in scene
     * @post Exception guarantee: No guarantee
     */
    SceneItem* getObjectInScene(std::shared_ptr<Course::BaseObject> obj);


    /**
     * @brief Returns the tile point of the mouse pointer
     * @param event points to the event that happened on the scene
     * @return Tile location of the mouse pointer in QPointF
     * @post Exception guarantee: No guarantee
     */
    QPointF mousePoint(QEvent* event);

    /**
     * @brief Receives events that happen on the game scene such as mouse actions
     *        and does required actions dependening on the event
     * @param event points to the event that happened on the scene
     * @return True: the event was  handled in the handler.
     *         False: the event handling was passed over.
     */
    virtual bool event(QEvent* event) override;



public slots:
    /**
     * @brief Is called by the timer to change the frame of every animated
     *        obejct.
     * @post Exception guarantee: No guarantee
     */
    void changeFrameForSceneItems();

private:
    QPointF lastClickPoint_;
    QPointF lastMousePoint_;

    std::weak_ptr<Course::iObjectManager> objectManager_;
    std::weak_ptr<iMenuObjectManager> menuObjectManager_;
    std::weak_ptr<Course::iGameEventHandler> eventHandler_;
    std::weak_ptr<GameSettingsManager> gameSettingsManager_;

    int mapGridSize_;
    int menuGridSize_;
    int width_;
    int height_;
    std::vector<std::string> mousePicture_;
    std::shared_ptr<FreeSceneItem> mouseDragItem_;
    QRectF mapBoundRect_;
    std::vector<std::shared_ptr<SceneItem>> sceneItems_;

    QTimer* timer_;
};

}

#endif // SIMPLEGAMESCENE_H

