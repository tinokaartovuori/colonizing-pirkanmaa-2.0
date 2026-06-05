/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: ipressableobject.h, interface for pressable objects          #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef IPRESSABLEOBJECT_H
#define IPRESSABLEOBJECT_H

#include <memory>
#include <vector>

namespace Student {

class iPressableObject
{
public:

    virtual void clickAction() = 0;

};

}


#endif // IPRESSABLEOBJECT_H


