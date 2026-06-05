/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: freesceneitem.h, contains different vectors that that have   #
#                        file paths to image files                   #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MENUSCENEITEM_H
#define MENUSCENEITEM_H

#include <QGraphicsItem>
#include <QPainter>

#include <memory>
#include <map>

#include "Core/menuobject.h"
#include "Core/coordinate.h"
#include "Menus/menuview.h"
#include "Menus/menuobjectcontainer.h"
#include "Menus/label.h"
#include "Graphics/animationoption.h"
#include "Overlays/mousehoverborder.h"
#include "Graphics/sceneitem.h"


namespace Student {


class MenuSceneItem : public SceneItem
{
public:
    /**
     * @brief Constructor
     * @param obj shared_ptr to the obj.
     * @param size of the created item in pixels.
     * @pre obj must have a valid Coordinate.
     */

    MenuSceneItem(const std::shared_ptr<Student::MenuObject> &obj,
                  std::shared_ptr<Student::MenuView>& mv);

    MenuSceneItem(const std::shared_ptr<MenuObject> &obj,
                  std::shared_ptr<MenuObjectContainer> &mv);

    MenuSceneItem(const std::shared_ptr<MenuObject> &obj);

    MenuSceneItem(const std::shared_ptr<Student::MenuView> &obj);


    QRectF boundingRect() const override;

    std::string getType();

    void updateLoc();

    void paint(QPainter *painter, const QStyleOptionGraphicsItem *option, QWidget *widget);


    void setUpperLayer(std::shared_ptr<Student::MenuView> mv);

    void setUpperLayer(std::shared_ptr<Student::MenuObjectContainer> mv);


    void setText();

protected:

    QPoint absolute_coordinates = QPoint(0, 0);
    QPoint relative_coordinates = QPoint(0, 0);
    std::shared_ptr<Student::MenuObject> m_upperLayer;

    std::string m_text;
    int m_fontSize;
    QColor m_color;
    std::string m_style;


};

}
#endif // SceneItem_H
