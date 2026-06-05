/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: mapsceneitem.h, header for MapSceneItem-class                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MAPSCENEITEM_H
#define MAPSCENEITEM_H

#include <QGraphicsItem>
#include <QPainter>

#include <memory>
#include <map>

#include "Core/gameobject.h"
#include "Tiles/tilebase.h"
#include "Graphics/animationoption.h"
#include "Overlays/mousehoverborder.h"
#include "Graphics/sceneitem.h"

namespace Student {

/**
 * @brief The MapSceneItem class is derived from SceneItem.
 *        The class is used to draw tiles, buildings and overlays.
 */
class MapSceneItem : public SceneItem
{
public:


    /**
     * @brief MapSceneItem constructor
     * @param obj points to the object to be drawn
     */
    MapSceneItem(const std::shared_ptr<Course::GameObject> &obj);


    /**
     * @brief Returns the rectangle that is surrounding the item
     * @return QRectF
     */
    QRectF boundingRect() const override;


    /**
     * @brief Returns the type of the item. In this case it's "MapSceneItem"
     * @return String of the item's type
     * @post Exception guarantee: No-throw
     */
    std::string getType();


    /**
     * @brief Paints the items on the scene. The function is derived from the
     *        QGraphicsItem
     * @param painter points to the painter that paints on the scene
     * @param option is unused
     * @param widget is unused
     */
    void paint(QPainter *painter,
               const QStyleOptionGraphicsItem *option,
               QWidget *widget);


    /**
     * @brief Updates the item's coordinates
     */
    void updateLoc();

    


    
private:

    QPoint coordinatesAsQPoint_;
    

};

}
#endif // SceneItem_H
