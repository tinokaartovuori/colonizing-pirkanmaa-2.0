/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: mapsceneitem.cpp, see mapsceneitem.h for more info           #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "mapsceneitem.h"

#include <QDebug>
#include <iostream>


namespace Student {

MapSceneItem::MapSceneItem(const std::shared_ptr<Course::GameObject> &obj):
    SceneItem(obj)
{
    coordinatesAsQPoint_ = obj->getCoordinatePtr()->asQpoint();
}


QRectF MapSceneItem::boundingRect() const
{
    const QPoint& topleft = QPoint(coordinatesAsQPoint_.x() * width_,
                                   coordinatesAsQPoint_.y() * height_);
    const QPoint& bottomright = topleft + QPoint(width_, height_);
    return QRectF(topleft, bottomright);
}

std::string MapSceneItem::getType()
{
    return "MapSceneItem";
}


void MapSceneItem::paint(QPainter *painter,
                    const QStyleOptionGraphicsItem *option,
                    QWidget *widget)
{
    Q_UNUSED( option ); Q_UNUSED( widget );

    setAcceptHoverEvents(true);

    if (getBoundObject()->getType() == "MouseHoverBorder" )
    {
        painter->setOpacity(0.5);
    }

    /*If the pixmap is updated into smaller vector, this ensures that a frame that
     *doesn't exist in the new smaller pixmap doesn't get drawn. Otherwise, an empty
     *image would be drawn for one animation frame. */
    if (currentImageFrame_ > (int)itemPixmap_.size() &&
            itemPixmap_.size()>=1) {
        currentImageFrame_ = 1;
    }

    //Paints the mapsceneitem
    if (currentImageFrame_ <= (int)itemPixmap_.size()) {
        painter->drawPixmap(coordinatesAsQPoint_.x() * width_,
                            coordinatesAsQPoint_.y() * height_,
                            width_, height_,
                            itemPixmap_.at(currentImageFrame_ - 1));
    }


    //Draws the borders showing the owner of the tile
    if (std::dynamic_pointer_cast<Course::TileBase>(baseObject_) != nullptr) {

        std::shared_ptr<Course::TileBase> tile =
                std::dynamic_pointer_cast<Course::TileBase>(baseObject_);
        std::vector<QPixmap> borderPixmaps = tile->getOwnerBorderPixmap();

        if (borderPixmaps.size() != 0) {
            for (int i=0; i < (int)borderPixmaps.size(); ++i) {

                painter->setOpacity(0.6);
                painter->drawPixmap(coordinatesAsQPoint_.x() * width_,
                                    coordinatesAsQPoint_.y() * height_,
                                    width_, height_, borderPixmaps.at(i));
            }
        }
    }
}


void MapSceneItem::updateLoc()
{
    coordinatesAsQPoint_ = getBoundObject()->getCoordinate().asQpoint();
}

}


