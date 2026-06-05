/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: freesceneitem.cpp, see freesceneitem.h for more info     #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "freesceneitem.h"

#include <QDebug>
#include <iostream>
#include <algorithm>


namespace Student {

FreeSceneItem::FreeSceneItem(std::vector<std::string> imagevector, Student::AnimationOption ani, int x, int y, int width, int height):
    currentImageIndex_(1),
    animationDirection_(1),
    animationOption_(ani),
    width_(width),
    height_(height),
    coordinates_(QPoint(x, y)),
    itemPathVector_(imagevector)
{
    if (animationOption_.startRandomFrame()) {
        randomizeStart_ = true;
    }
    setItemPixmap();
}

QRectF FreeSceneItem::boundingRect() const {
    return QRectF(QPoint(coordinates_.x(), coordinates_.y()), QPoint(width_, height_));
}

void FreeSceneItem::paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget)
{
    Q_UNUSED( option ); Q_UNUSED( widget );

    painter->drawPixmap(coordinates_.x(),
                        coordinates_.y(),
                        width_, height_,
                        itemPixmap_.at(currentImageIndex_ - 1));
}

void FreeSceneItem::updateLoc(int x, int y)
{

    coordinates_ = QPoint(x, y);

}

void FreeSceneItem::setItemPixmap()
{
    std::vector<std::string> v = itemPathVector_;
    for (const auto& item : v) {
        QString filename = QString::fromStdString(item);
        QPixmap pix(filename);
        itemPixmap_.push_back(pix);
    }

}


void FreeSceneItem::setRandomImageIndex()
{
    currentImageIndex_ = (rand() % itemPixmap_.size());
}

void FreeSceneItem::changeAnimationFrame()
{
    if (animationOption_.isAnimated() == false) return;
    if (randomizeStart_) {
        setRandomImageIndex();
        randomizeStart_ = false;
    }

    currentImageIndex_ += animationDirection_;

    int amount_of_images = itemPixmap_.size();

    if (animationOption_.getStyle() == "rollover") {

        if (currentImageIndex_ >= amount_of_images + 1){
            currentImageIndex_ = 1;
        }
    }

    if (animationOption_.getStyle() == "backandforth") {

        if (currentImageIndex_ <= 1) {
            animationDirection_ = 1;
            currentImageIndex_ = 1;
        }
        if (currentImageIndex_ >= amount_of_images) {
            currentImageIndex_ = amount_of_images;
            animationDirection_ = -1;
        }
    }
}


int FreeSceneItem::getHeight()
{
    return height_;
}


int FreeSceneItem::getWidth()
{
    return width_;
}


void FreeSceneItem::setWidth(int width)
{
    width_ = width;
}


void FreeSceneItem::setHeight(int height)
{
    height_ = height;
}

std::string FreeSceneItem::getType() {
    return "FreeSceneItem";
}

} //namespace Course
