/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: moushoverborder.h, header for MouseHoverBorder class         #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MOUSEHOVERBORDER_H
#define MOUSEHOVERBORDER_H


#include "Core/gameobject.h"


namespace Student {

/**
 * @brief Object that is on the tile the mouse is pointing to
 */
class MouseHoverBorder : public Course::GameObject
{
public:
    MouseHoverBorder(const Course::Coordinate& coordinate,
             int width,
             int height,
             const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
             const std::weak_ptr<Course::iObjectManager>& objectmanager);

    virtual std::string getType() const override;

    bool drawn();

    void setDrawn(bool d);

    virtual void clickAction();

private:

    bool m_drawn;

};
}


#endif // MOUSEHOVERBORDER_H
