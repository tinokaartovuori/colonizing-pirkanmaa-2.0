/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: freesceneitem.h, header for FreeSceneItem-class              #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef FREESCENEITEM_H
#define FREESCENEITEM_H

#include <QGraphicsItem>
#include <QPainter>

#include <memory>
#include <map>

#include "Core/gameobject.h"
#include "Graphics/animationoption.h"
#include "Overlays/mousehoverborder.h"

namespace Student {

/**
 * @brief Class used to draw items that can move freely so they are not in
 *        any grid
 */

class FreeSceneItem : public QGraphicsItem
{
public:

    FreeSceneItem(std::vector<std::string> imagevector,
                  Student::AnimationOption ani,
                  int x,
                  int y,
                  int width,
                  int height);


    virtual QRectF boundingRect() const override;

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget) override;

    void setItemPixmap();

    void setRandomImageIndex();

    void changeAnimationFrame();

    int getWidth();

    int getHeight();

    void setWidth(int width);

    void setHeight(int height);

    virtual std::string getType();

    void updateLoc(int x, int y);

protected:

    int currentImageIndex_;
    int animationDirection_;
    Student::AnimationOption animationOption_;

    int width_;
    int height_;
    QPoint coordinates_;
    std::vector<std::string> itemPathVector_;

    std::vector<QPixmap> itemPixmap_;

    bool randomizeStart_ = false;
};

}
#endif // FREESCENEITEM_H
