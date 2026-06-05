/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: unitsceneitem.h, header for UnitSceneItem-class              #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef UNITSCENEITEM_H
#define UNITSCENEITEM_H

#include <QGraphicsItem>
#include <QPainter>

#include <memory>
#include <map>

#include "Units/unitbase.h"
#include "Tiles/tilebase.h"
#include "Core/coordinate.h"

#include "Graphics/animationoption.h"
#include "Graphics/sceneitem.h"


namespace Student {

/**
 * @brief The MapSceneItem class is derived from SceneItem.
 *        The class is used to draw units. The units have their own tile
 *        specific
 */
class UnitSceneItem : public SceneItem
{
public:


    UnitSceneItem(const std::shared_ptr<Course::UnitBase> &obj);

    QRectF boundingRect() const override;

    std::string getType();

    void updateLoc();

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);


protected:

    QPoint absolute_coordinates = QPoint(0, 0);
    QPoint relative_coordinates = QPoint(0, 0);

};

}
#endif // SceneItem_H
