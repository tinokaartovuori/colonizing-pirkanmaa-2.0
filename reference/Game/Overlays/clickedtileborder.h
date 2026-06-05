/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: clickedtileborder.h, header for ClickedTileBorder class      #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef CLICKEDTILEBORDER_H
#define CLICKEDTILEBORDER_H


#include "Core/gameobject.h"


namespace Student {

/**
 * @brief Is used to show a box on the tile that was clicked
 */
class ClickedTileBorder : public Course::GameObject
{
public:
    ClickedTileBorder(const Course::Coordinate& coordinate,
             int width,
             int height,
             const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
             const std::weak_ptr<Course::iObjectManager>& objectmanager);

    virtual std::string getType() const override;

    virtual void clickAction();
};

} //namespace Student


#endif // CLICKEDTILEBORDER_H
