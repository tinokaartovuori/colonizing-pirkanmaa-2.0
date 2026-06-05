/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: icontainer.h, interface for MenuObject and MenuView-classes  #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef ICONTAINER_H
#define ICONTAINER_H

#include <memory>
#include <vector>

#include "Core/menuobject.h"

namespace Student {

class iContainer
{
public:

    virtual void addMenuObject(const std::shared_ptr<MenuObject>& obj) = 0;

    virtual std::vector<std::shared_ptr<MenuObject>> getMenuObjects() const = 0;


};

}


#endif // ICONTAINER_H


