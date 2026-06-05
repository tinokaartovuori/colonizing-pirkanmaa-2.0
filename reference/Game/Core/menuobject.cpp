/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuobject.cpp, see menuobject.h for more info               #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "menuobject.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Exceptions/keyerror.h"
#include "Exceptions/invalidpointer.h"
#include "playerbase.h"

#include <QDebug>


#include <algorithm>

namespace Student {

MenuObject::MenuObject(const Course::Coordinate& coordinate,
                       const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
                       const std::weak_ptr<Course::iObjectManager>& objectmanager):
    BaseObject(coordinate,  eventhandler, objectmanager)
{
    isMultiPixMap_ = false;
}

MenuObject::MenuObject(const Course::Coordinate& coordinate,
                       int width,
                       int height,
                       const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
                       const std::weak_ptr<Course::iObjectManager>& objectmanager):
    BaseObject(coordinate, width, height, eventhandler, objectmanager)
{
    isMultiPixMap_ = false;
    isInverseMultiPixMap_ = false;
}

std::string MenuObject::getType() const
{
    return "MenuObject";
}

void MenuObject::addToAbsoluteCoordinate(QPoint coord) {
    absoluteCoordinate += coord;
}

QPoint MenuObject::getAbsoluteCoordinates() {
    return absoluteCoordinate;
}

void MenuObject::multiPixMap(bool onoff) {
    isMultiPixMap_ = onoff;
}

bool MenuObject::isMultiPixMap()
{
    return isMultiPixMap_;
}

void MenuObject::inverseMultiPixMap(bool onoff) {
    isInverseMultiPixMap_ = onoff;
}

bool MenuObject::isInverseMultiPixMap() {
    return isInverseMultiPixMap_;
}


}

