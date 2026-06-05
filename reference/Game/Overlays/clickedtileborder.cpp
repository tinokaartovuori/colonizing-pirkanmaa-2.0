/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: clicktileborder.cpp, see clicktileborder.h for the class's description #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "clickedtileborder.h"
#include <qdebug.h>


namespace Student {

ClickedTileBorder::ClickedTileBorder(const Course::Coordinate &coordinate,
        int width,
        int height,
        const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
        const std::weak_ptr<Course::iObjectManager> &objectmanager):
    Course::GameObject(coordinate, width, height, eventhandler, objectmanager)
{
}

std::string ClickedTileBorder::getType() const
{
    return "ClickedTileBorder";
}

void ClickedTileBorder::clickAction()
{
    qDebug() << "ClickedTileBorder";
}

} //namespace Course
