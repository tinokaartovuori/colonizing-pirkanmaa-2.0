/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: moushoverborder.cpp, see moushoverborder.h for the class's description #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "mousehoverborder.h"
#include <qdebug.h>


namespace Student {

MouseHoverBorder::MouseHoverBorder(
        const Course::Coordinate &coordinate,
        int width,
        int height,
        const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
        const std::weak_ptr<Course::iObjectManager> &objectmanager):
    Course::GameObject(coordinate, width, height, eventhandler, objectmanager)
{
    m_drawn = false;
}

std::string MouseHoverBorder::getType() const
{
    return "MouseHoverBorder";
}



bool MouseHoverBorder::drawn()
{
    return m_drawn;
}

void MouseHoverBorder::setDrawn(bool d)
{
    m_drawn = d;
}

void MouseHoverBorder::clickAction()
{
    qDebug() << "MouseHoverBorder";
}


} //namespace Course
