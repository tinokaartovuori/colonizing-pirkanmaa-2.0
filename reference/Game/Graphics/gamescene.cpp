/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: gamescene.cpp, see gamescene.h for more info                 #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include <QEvent>
#include <QHoverEvent>
#include <QGraphicsSceneMouseEvent>
#include <QDebug>
#include <QtGlobal>
#include <typeinfo>

#include <math.h>

#include "gamescene.h"
#include "sceneitem.h"
#include "mapsceneitem.h"
#include "menusceneitem.h"
#include "unitsceneitem.h"


namespace Student {

GameScene::GameScene(QWidget* parent,
               std::shared_ptr<Course::iGameEventHandler> eventHandler,
               std::shared_ptr<Course::iObjectManager> objectManager,
               std::shared_ptr<iMenuObjectManager> menuObjectManager,
               std::shared_ptr<GameSettingsManager> gameSettingsManager):
    QGraphicsScene(parent),
    lastClickPoint_(-1, -1),
    lastMousePoint_(-1, -1),

    objectManager_(objectManager),
    menuObjectManager_(menuObjectManager),
    eventHandler_(eventHandler),
    gameSettingsManager_(gameSettingsManager),

    mapGridSize_(gameSettingsManager->getMapGridSize()),
    menuGridSize_(gameSettingsManager->getMenuGridSize()),
    width_(gameSettingsManager->getMapGridWidth()),
    height_(gameSettingsManager->getMapGridHeight()),
    mousePicture_(),
    mouseDragItem_(),
    mapBoundRect_(QRectF(QPointF(0, 0),
                          QPointF(gameSettingsManager->getMapGridWidth() - 1,
                                  gameSettingsManager->getMapGridHeight() - 1))),
    sceneItems_()
{
    timer_ = new QTimer(this);
    QObject::connect(timer_, SIGNAL(timeout()),
                     this, SLOT(changeFrameForSceneItems()));
    timer_->start(450);
}


void GameScene::deleteObjects()
{
    for (auto item : sceneItems_) {
        item.reset();
    }

    if (timer_ != nullptr) {
        delete timer_;
    }
}


void GameScene::addSceneItem(std::shared_ptr<SceneItem> sceneItem)
{
    sceneItems_.push_back(sceneItem);
}


void GameScene::drawItem(std::shared_ptr<Course::GameObject> obj)
{
    std::shared_ptr<Student::MapSceneItem> nItem =
            std::make_shared<Student::MapSceneItem>(obj);
    if (std::dynamic_pointer_cast<Course::BuildingBase>(obj) != nullptr) {
        nItem->setZValue(1);
    }
    if (std::dynamic_pointer_cast<Student::ClickedTileBorder>(obj) != nullptr) {
        nItem->setZValue(10);
    }
    if (std::dynamic_pointer_cast<Student::MouseHoverBorder>(obj) != nullptr) {
        nItem->setZValue(10);
    }
    if (std::dynamic_pointer_cast<Student::BlockedTile>(obj) != nullptr) {
        nItem->setZValue(8);
    }
    nItem->setGridSize(mapGridSize_);
    addSceneItem(nItem);

    nItem->setItemPixmap(); //Creates a vector in the item of all the frames
    addItem(nItem.get());

    nItem->setWidth(obj->getWidth()*mapGridSize_);
    nItem->setHeight(obj->getHeight()*mapGridSize_);

    emit updateScene();
}


void GameScene::drawItem(std::shared_ptr<Course::UnitBase> obj)
{
    std::shared_ptr<Student::UnitSceneItem> nItem =
            std::make_shared<Student::UnitSceneItem>(obj);
    nItem->setZValue(3);
    addSceneItem(nItem);
    nItem->setItemPixmap(); //Creates a vector in the item of all the frames
    addItem(nItem.get());

    nItem->setWidth(obj->getWidth() * mapGridSize_);
    nItem->setHeight(obj->getHeight() * mapGridSize_);

    emit updateScene();
}


void GameScene::drawItem(std::shared_ptr<Student::MenuView> obj)
{
    bool already = false;
    for (std::vector<std::shared_ptr<SceneItem>>::iterator it =
         sceneItems_.begin(); it != sceneItems_.end();)
    {
        if (obj->getType() == "MenuView") {
            already = true;
            break;
        }

    }
    if (!already) {
        std::shared_ptr<Student::MenuSceneItem> nItem =
                std::make_shared<Student::MenuSceneItem>(obj);
        nItem->setGridSize(menuGridSize_);
        addSceneItem(nItem);
        nItem->setItemPixmap(); //Creates a vector in the item of all the frames
        addItem(nItem.get());

        nItem->setWidth(obj->getWidth() * menuGridSize_);
        nItem->setHeight(obj->getHeight() * menuGridSize_);
    }

    try {

        for (auto o : obj->getMenuObjects()) {
            if (o->getType() == "MenuObjectContainer") {
                std::shared_ptr<Student::MenuObjectContainer> con =
                        std::static_pointer_cast<Student::MenuObjectContainer>(o);
                drawItem(con, obj);
            }
            else {
                drawItem(o, obj);
            }
        }
    }
    catch (...) {
        qDebug() << "Empty Container";
    }

    emit updateScene();
}


void GameScene::drawItem(std::shared_ptr<Student::MenuObjectContainer> obj,
                         std::shared_ptr<Student::iContainer> cont)
{

    std::shared_ptr<Student::MenuSceneItem> nItem;
    if (std::dynamic_pointer_cast<MenuObjectContainer>(cont) != nullptr) {
        std::shared_ptr<Student::MenuObjectContainer> container =
                std::dynamic_pointer_cast<MenuObjectContainer>(cont);
        nItem = std::make_shared<MenuSceneItem>(obj, container);
    }
    else if (std::dynamic_pointer_cast<MenuView>(cont) != nullptr) {
        std::shared_ptr<Student::MenuView> container =
                std::dynamic_pointer_cast<MenuView>(cont);
        nItem = std::make_shared<MenuSceneItem>(obj, container);
    }

    nItem->setGridSize(menuGridSize_);
    addSceneItem(nItem);
    nItem->setItemPixmap(); //Creates a vector in the item of all the frames
    addItem(nItem.get());
    nItem->setWidth(obj->getWidth()*menuGridSize_);
    nItem->setHeight(obj->getHeight()*menuGridSize_);

    try {
        for (auto o : obj->getMenuObjects()) {
            if (o->getType() == "MenuObjectContainer") {
                std::shared_ptr<Student::MenuObjectContainer> con =
                        std::static_pointer_cast<Student::MenuObjectContainer>(o);
                drawItem(con, obj);
            }
            else {
                drawItem(o, obj);
            }
        }
    }
    catch (...) {
        qDebug() << "Empty Container";
    }


    emit updateScene();
}


void GameScene::drawItem(std::shared_ptr<Student::MenuObject> obj,
                         std::shared_ptr<Student::iContainer> cont)
{

    std::shared_ptr<Student::MenuSceneItem> nItem;
    if (std::dynamic_pointer_cast<Student::MenuObjectContainer>(cont) != nullptr)
    {
        std::shared_ptr<Student::MenuObjectContainer> container =
                std::dynamic_pointer_cast<Student::MenuObjectContainer>(cont);

        nItem = std::make_shared<Student::MenuSceneItem>(obj, container);
    }
    else if (std::dynamic_pointer_cast<Student::MenuView>(cont) != nullptr)
    {
        std::shared_ptr<Student::MenuView> container =
                      std::dynamic_pointer_cast<Student::MenuView>(cont);
        nItem = std::make_shared<Student::MenuSceneItem>(obj, container);
    }

    nItem->setGridSize(menuGridSize_);
    addSceneItem(nItem);


    if (std::dynamic_pointer_cast<Student::iLabel>(obj) != nullptr) {
        nItem->setText();
    }
    if (obj->getType() != "Label") {
        nItem->setItemPixmap(); //Creates a vector in the item of all the frames
    }

    addItem(nItem.get());
    nItem->setWidth(obj->getWidth()*menuGridSize_);
    nItem->setHeight(obj->getHeight()*menuGridSize_);

    emit updateScene();
}


void GameScene::drawMouseFollowItem(QEvent *event)
{
    QGraphicsSceneMouseEvent* mouse_event =
                dynamic_cast<QGraphicsSceneMouseEvent*>(event);
    QPointF abs_point = mouse_event->scenePos();
    int x = abs_point.x();
    int y = abs_point.y();

    if (mousePicture_.size() != 0) {
        std::shared_ptr<Student::FreeSceneItem> nItem =
                std::make_shared<Student::FreeSceneItem>
                (mousePicture_, AnimationOptions::UNIT, x-10, y-15, 20, 30);

        mouseDragItem_ = nItem;
        mouseDragItem_->setZValue(11);
        mouseDragItem_->setItemPixmap();
        addItem(mouseDragItem_.get());
    }
    else {
        mouseDragItem_.reset();
    }

    emit updateScene();
}


void GameScene::updateItem(std::shared_ptr<Course::GameObject> obj)
{
    for (auto item : items()){
        Student::MapSceneItem* mapsceneitem =
                            static_cast<Student::MapSceneItem*>(item);

        if (mapsceneitem->isSameObj(obj)){
            mapsceneitem->setItemPixmap();
            mapsceneitem->updateLoc();
            emit updateScene();
        }
    }
}


void GameScene::updateItem(std::shared_ptr<Student::MenuObject> obj)
{
    for (auto item : items()){
        Student::MenuSceneItem* menusceneitem =
                     static_cast<Student::MenuSceneItem*>(item);

        if (menusceneitem->isSameObj(obj)){
            menusceneitem->updateLoc();
            emit updateScene();
        }
    }
}


void GameScene::updateTile(std::shared_ptr<Course::TileBase> tileobj)
{
    QList<QGraphicsItem*> items_list = items();

    if (tileobj->getBuilding() != nullptr) {
        bool hasntBeenDrawn = true;
        for (auto item : items_list) {
            if (static_cast<Student::MapSceneItem*>(item) == nullptr) continue;
            Student::MapSceneItem* mapsceneitem =
                                static_cast<Student::MapSceneItem*>(item);
            /*Building has already been drawn so it is updated in
             *case if the graphics have been changed.*/
            if (mapsceneitem->isSameObj(tileobj->getBuilding())){
                updateItem(tileobj->getBuilding());
                hasntBeenDrawn = false;
            }

        }
        if (hasntBeenDrawn) {
            drawItem(tileobj->getBuilding());
        }

    }

    for (auto unit : tileobj->getUnits()) {
        removeItem(unit);
    }

    if (tileobj->getUnitCount() > 0) {
        for (auto unit : tileobj->getUnits()) {
            drawItem(unit);
        }
    }

    for (auto unit : tileobj->getConqueringUnits()) {
        removeItem(unit);
    }

    if (tileobj->getConqueringUnitCount() > 0) {
        for (auto unit : tileobj->getConqueringUnits()) {
            drawItem(unit);
        }
    }

}


void GameScene::removeItem(std::shared_ptr<Course::BaseObject> obj)
{
    for (std::vector<std::shared_ptr<SceneItem>>::iterator it =
                            sceneItems_.begin(); it != sceneItems_.end();)
    {
        if (obj->getType() == "MenuView") continue;
        if ((*it)->isSameObj(obj)){
            it = sceneItems_.erase(it);
            emit updateScene();
        } else {
            ++it;
        }
    }
}


void GameScene::removeContainer(std::shared_ptr<Student::iContainer> cont)
{

    if (cont->getMenuObjects().size() > 0) {
        for (auto obj : cont->getMenuObjects()) {
            if (obj->getType() == "MenuObjectContainer") {

                std::shared_ptr<Student::MenuObjectContainer> con =
                        std::static_pointer_cast<Student::MenuObjectContainer>(obj);
                removeContainer(con);
                continue;
            }

            if ( sceneItems_.size() == 1 ){
            } else {
                for(std::vector<std::shared_ptr<SceneItem>>::iterator it =
                    sceneItems_.begin(); it != sceneItems_.end();)
                {
                    if (obj->getType() == "MenuView") continue;
                    if ((*it)->isSameObj(obj)){
                        it = sceneItems_.erase(it);
                        emit updateScene();
                    } else {
                        ++it;
                    }
                }
            }
        }
    }

    std::shared_ptr<Student::MenuObject> con =
            std::dynamic_pointer_cast<Student::MenuObject>(cont);
    if (con->getType() == "MenuObjectContainer") {
        removeItem(con);
    }

}


void GameScene::removeMouseFollowItem()
{
    mousePicture_ = {};
}


void GameScene::addMouseFollowPicture(std::vector<std::string> imagevector)
{
    mousePicture_ = std::vector<std::string>(imagevector);
}


bool GameScene::isObjectInScene(std::shared_ptr<Course::BaseObject> obj)
{
    QList<QGraphicsItem*> items_list = items();
    if ( items_list.size() != 1 ){
        for (auto item : items_list){
            SceneItem* sceneitem = static_cast<SceneItem*>(item);
            if ( sceneitem->isSameObj(obj) ){
                return true;
            }
        }
    }
    return false;
}


SceneItem* GameScene::getObjectInScene(std::shared_ptr<Course::BaseObject> obj)
{
    QList<QGraphicsItem*> items_list = items();
    if ( items_list.size() != 1 ) {
        for (auto item : items_list) {
            SceneItem* sceneitem = static_cast<SceneItem*>(item);
            if ( sceneitem->isSameObj(obj) ){
                return sceneitem;
            }
        }
    }
    return nullptr;
}


QPointF GameScene::mousePoint(QEvent *event)
{
    QGraphicsSceneMouseEvent* mouse_event =
                dynamic_cast<QGraphicsSceneMouseEvent*>(event);
        QPointF point = mouse_event->scenePos() / mapGridSize_;
        point.rx() = floor(point.rx());
        point.ry() = floor(point.ry());

        return point;

}


bool GameScene::event(QEvent *event)
{
    /*A mechanism that is used to draw the border that is on the
     *tile the mouse cursor is pointing to. The border cannot be
     *drawn at this stage since the mouse coordinates are unknown
     *when QEvent::Enter happens.*/



    if (event->type() == QEvent::Enter) {
        objectManager_.lock()->getBorderTile()->setDrawn(false);
        lastMousePoint_ = QPoint(-1, -1);
        return true;
    }

    /* When mouse leaves the area the border that is on the tile the
     * mouse cursor is pointing to is removed. */
    else if (event->type() == QEvent::Leave)
    {
        removeItem(objectManager_.lock()->getBorderTile());
        return true;
    }

    /*Draws or moves the border that is below the mouse cursor
     *with the help of the mouse coordinates. */
    else if (event->type() == QEvent::GraphicsSceneMouseMove) {

        QPointF point = mousePoint(event);

        drawMouseFollowItem(event);

        if (lastMousePoint_ == point) return true;
        lastMousePoint_ = point;


        if (!mapBoundRect_.contains(point)) {

            objectManager_.lock()->getBorderTile()->setDrawn(false);
            removeItem(objectManager_.lock()->getBorderTile());
            lastMousePoint_ = QPoint(-1, -1);
            return true;
        }

        if (!objectManager_.lock()->getBorderTile()->drawn()) {
            drawItem(objectManager_.lock()->getBorderTile());
        }

        objectManager_.lock()->getBorderTile()->setDrawn(true);

        for(unsigned i = sceneItems_.size() - 1; sceneItems_.size() > i; --i)
        {
            std::shared_ptr<SceneItem> sceneItem = sceneItems_.at(i);
            QRectF boundingRect_;

            if (sceneItem->getType() == "MapSceneItem") {
                std::shared_ptr<MapSceneItem> item = std::static_pointer_cast<MapSceneItem>(sceneItem);
                boundingRect_ = item->boundingRect();
            } else {

                continue;
            }


            if (floor(boundingRect_.x() / mapGridSize_) == point.x() and
                    floor(boundingRect_.y() / mapGridSize_) == point.y()) {

                objectManager_.lock()->getBorderTile()
                        ->setCoordinate(Course::Coordinate(point.x(), point.y()));

                updateItem(objectManager_.lock()->getBorderTile());

                break;
            }
        }
        return true;
    }


    else if(event->type() == QEvent::GraphicsSceneMousePress) {
        //QPointF point = mousePoint(event);

        QGraphicsSceneMouseEvent* mouse_event =
                dynamic_cast<QGraphicsSceneMouseEvent*>(event);
        QPointF point = mouse_event->scenePos();

        for(unsigned i = sceneItems_.size() - 1; sceneItems_.size() > i; --i)
        {
            std::shared_ptr<SceneItem> sceneItem = sceneItems_.at(i);

            // Checking if BoundObject is derived from iPressableObject
            if (std::dynamic_pointer_cast<Student::iPressableObject>(sceneItem->getBoundObject()) == nullptr) continue;

            QRectF boundingRect_;

            if (sceneItem->getType() == "MenuSceneItem") {
                std::shared_ptr<Student::MenuSceneItem> item = std::static_pointer_cast<Student::MenuSceneItem>(sceneItem);
                boundingRect_ = item->boundingRect();
            }
            else if (sceneItem->getType() == "MapSceneItem") {
                std::shared_ptr<Student::MapSceneItem> item = std::static_pointer_cast<Student::MapSceneItem>(sceneItem);
                boundingRect_ = item->boundingRect();
            }
            else if (sceneItem->getType() == "UnitSceneItem") {
                std::shared_ptr<Student::UnitSceneItem> item = std::static_pointer_cast<Student::UnitSceneItem>(sceneItem);
                boundingRect_ = item->boundingRect();
            }

            if (boundingRect_.contains(point)) {
                std::dynamic_pointer_cast<Student::iPressableObject>(sceneItem->getBoundObject())->clickAction();
                drawMouseFollowItem(event);
                break;
            }
        }
        return true;
    }

   return false;
}


void GameScene::changeFrameForSceneItems()
{
    QList<QGraphicsItem*> items_list = items();
    for (auto item : items_list){
        SceneItem* sceneitem = static_cast<SceneItem*>(item);
        sceneitem->changeAnimationFrame();
    }
    emit updateScene();
}



}
