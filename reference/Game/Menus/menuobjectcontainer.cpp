/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuobjectcontainer.cpp                                      #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "menuobjectcontainer.h"

#include <QtGlobal> // For Q_ASSERT
#include <QDebug>

#include "Exceptions/notenoughspace.h"
#include "Exceptions/ownerconflict.h"
#include "Exceptions/invalidpointer.h"
#include "Core/playerbase.h"


namespace Student {

MenuObjectContainer::MenuObjectContainer(const Course::Coordinate& coordinate,
                   int width,
                   int height,
                   int gridSize,
                   const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                   const std::weak_ptr<Course::iObjectManager>& objectmanager):
    MenuObject(coordinate, width, height, eventhandler, objectmanager),
    m_upperLayer({}),
    m_gridSize(gridSize)
{
    absoluteCoordinate += coordinate.asQpoint() * gridSize;
}

MenuObjectContainer::MenuObjectContainer(const Course::Coordinate &coordinate,
                   const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                   const std::weak_ptr<Course::iObjectManager> &objectmanager):
    MenuObject(coordinate, eventhandler, objectmanager)
{
}

std::string MenuObjectContainer::getType() const
{
    return "MenuObjectContainer";
}

void MenuObjectContainer::addMenuObject(const std::shared_ptr<MenuObject> &obj) {

    m_upperLayer.push_back(obj);
    obj->addToAbsoluteCoordinate(getAbsoluteCoordinates());

}


std::vector<std::shared_ptr<MenuObject>> MenuObjectContainer::getMenuObjects() const
{
    return m_upperLayer;
}

int MenuObjectContainer::getGridSize() const
{
    return m_gridSize;
}



} // namespace Student
