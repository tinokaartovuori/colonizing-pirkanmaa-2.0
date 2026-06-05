/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: unitsceneitem.cpp, see unitsceneitem.h for more info         #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "unitsceneitem.h"

#include <QDebug>
#include <iostream>
#include <math.h>

namespace Student {


UnitSceneItem::UnitSceneItem(const std::shared_ptr<Course::UnitBase> &obj):
    SceneItem(obj)
{
    gridSize_ = obj->getGridSize();
    relative_coordinates = obj->getTileRelatedCoordinates()->asQpoint() * round(obj->getGridSize() / 6);
    absolute_coordinates = obj->getParentTile()->getCoordinate().asQpoint() * obj->getGridSize();

}


QRectF UnitSceneItem::boundingRect() const
{
    return QRectF(absolute_coordinates + relative_coordinates * 2,
                  absolute_coordinates + relative_coordinates * 3
                  + QPoint(width_ * 2/6, height_ * 3/6));
}


std::string UnitSceneItem::getType()
{
    return "UnitSceneItem";
}


void UnitSceneItem::paint(QPainter *painter,
                    const QStyleOptionGraphicsItem *option,
                    QWidget *widget)
{
    Q_UNUSED( option ); Q_UNUSED( widget );

    setAcceptHoverEvents(true);

    setZValue(3);

    if (itemPixmap_.size() > 0) {
        painter->drawPixmap(absolute_coordinates.x() + (relative_coordinates.x() * 2),
                            absolute_coordinates.y() + (relative_coordinates.y() * 3),
                            width_ * 2/6, height_ * 3/6,
                            itemPixmap_.at(currentImageFrame_ - 1));
    }

}


void UnitSceneItem::updateLoc()
{
    if ( !baseObject_ )
    {
        delete this;
    }
    else {
        relative_coordinates = baseObject_->getCoordinate().asQpoint();

    }
}


} //namespace Course


