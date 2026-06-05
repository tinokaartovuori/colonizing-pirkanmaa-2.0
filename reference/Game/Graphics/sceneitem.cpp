/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: sceneitem.cpp, see sceneitem.h for more info                 #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#include "sceneitem.h"

#include <QDebug>
#include <iostream>


namespace Student {

SceneItem::SceneItem(const std::shared_ptr<Course::BaseObject> &obj):
    baseObject_(obj),
    currentImageFrame_(1),
    animationDirection_(1),
    animationOption_(baseObject_->getAnimationOption()),
    itemPixmap_(),
    gridSize_(0)
{
    if (animationOption_.startRandomFrame()) {
        randomizeStart_ = true;
    }
    width_ = baseObject_->getWidth();
    height_ = baseObject_->getHeight();
}


const std::shared_ptr<Course::BaseObject> &SceneItem::getBoundObject()
{
    return baseObject_;
}


bool SceneItem::isSameObj(std::shared_ptr<Course::BaseObject> obj)
{
    return obj == baseObject_;
}


QRectF SceneItem::boundingRect() const {
    return QRectF(QPoint(0,0), QPoint(0,0));
}

void SceneItem::setItemPixmap()
{
    std::vector<std::string> v = baseObject_->getImageFiles();

    itemPixmap_ = {};

    for (const auto& item : v) {
        QString filename = QString::fromStdString(item);
        QPixmap pix(filename);
        itemPixmap_.push_back(pix);
    }

}


void SceneItem::setAnimationFrame(int frame)
{
    currentImageFrame_ = frame;
}


void SceneItem::setRandomImageIndex()
{
    currentImageFrame_ = (rand() % itemPixmap_.size());
}


void SceneItem::changeAnimationFrame()
{
    if (animationOption_.isAnimated() == false) return;
    if (randomizeStart_) {
        setRandomImageIndex();
        randomizeStart_ = false;
    }

    currentImageFrame_ += animationDirection_;

    int amount_of_images = itemPixmap_.size();

    if (animationOption_.getStyle() == "rollover") {

        if (currentImageFrame_ >= amount_of_images + 1){
            currentImageFrame_ = 1;
        }
    }

    if (animationOption_.getStyle() == "backandforth") {

        if (currentImageFrame_ <= 1) {
            animationDirection_ = 1;
            currentImageFrame_ = 1;
        }
        if (currentImageFrame_ >= amount_of_images) {
            currentImageFrame_ = amount_of_images;
            animationDirection_ = -1;
        }
    }
}


void SceneItem::setGridSize(int gridSize)
{
    gridSize_ = gridSize;
}


void SceneItem::setWidth(int width)
{
    width_ = width;
}


void SceneItem::setHeight(int height)
{
    height_ = height;
}


std::string SceneItem::getType() {
    return "SceneItem";
}


void SceneItem::setAnimationOption(AnimationOption ani)
{
    animationOption_ = ani;
}


} //namespace Student
