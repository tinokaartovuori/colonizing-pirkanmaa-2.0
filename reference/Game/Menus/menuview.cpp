/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuview.cpp                                                 #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "menuview.h"

#include <QtGlobal> // For Q_ASSERT
#include <QDebug>

#include "Exceptions/notenoughspace.h"
#include "Exceptions/ownerconflict.h"
#include "Exceptions/invalidpointer.h"
#include "Core/playerbase.h"


namespace Student {

MenuView::MenuView(const Course::Coordinate& coordinate,
                   int width,
                   int height,
                   int gridSize,
                   const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                   const std::weak_ptr<Course::iObjectManager>& objectmanager):
    MenuObject(coordinate, width, height, eventhandler, objectmanager), m_gridSize(gridSize)
{
    absoluteCoordinate = coordinate.asQpoint();
    setImageFiles(ImageVectors::MULTI);
    multiPixMap(true);
}


MenuView::MenuView(const Course::Coordinate &coordinate,
                   const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                   const std::weak_ptr<Course::iObjectManager> &objectmanager):
    MenuObject(coordinate, eventhandler, objectmanager)
{
    absoluteCoordinate = coordinate.asQpoint();
}


std::string MenuView::getType() const
{
    return "MenuView";
}

void MenuView::addMenuObject(const std::shared_ptr<MenuObject> &obj) {

    m_upperLayer.push_back(obj);
    obj->addToAbsoluteCoordinate(getAbsoluteCoordinates());
}


std::vector<std::shared_ptr<MenuObject>> MenuView::getMenuObjects() const
{
    return m_upperLayer;
}


int MenuView::getGridSize() const
{
    return m_gridSize;
}



} // namespace Student
