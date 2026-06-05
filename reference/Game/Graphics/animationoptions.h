/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: animationoptions.h contains different AnimationOption        #
#                          classes assigned into variables           #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef ANIMATIONOPTIONS_H
#define ANIMATIONOPTIONS_H

#include <vector>
#include <string>
#include "animationoption.h"


namespace AnimationOptions {

const Student::AnimationOption CLICKEDTILEBORDER =
        Student::AnimationOption(false);
const Student::AnimationOption MOUSEHOVERBORDER =
        Student::AnimationOption(true, "rollover");

const Student::AnimationOption FOREST =
        Student::AnimationOption(true, "backandforth", true);

const Student::AnimationOption GRASSLAND =
        Student::AnimationOption(false);

const Student::AnimationOption MIKONTALO =
        Student::AnimationOption(false);

const Student::AnimationOption MOUNTAIN =
        Student::AnimationOption(false);

const Student::AnimationOption MOUNTAIN_FOREST =
        Student::AnimationOption(true, "backandforth", true);

const Student::AnimationOption CONTAINER =
        Student::AnimationOption(false);

const Student::AnimationOption BUTTON =
        Student::AnimationOption(false);

const Student::AnimationOption MENU =
        Student::AnimationOption(false);

const Student::AnimationOption HEADQUARTERS =
        Student::AnimationOption(true, "rollover");

const Student::AnimationOption UNIT =
        Student::AnimationOption(true, "rollover");

const Student::AnimationOption NUCLEAR =
        Student::AnimationOption(true, "rollover");

const Student::AnimationOption HEPP =
        Student::AnimationOption(true, "backandforth");

const Student::AnimationOption OUTPOST =
        Student::AnimationOption(true, "backandforth");

const Student::AnimationOption EMPTY =
        Student::AnimationOption(false);

const Student::AnimationOption RIVER =
        Student::AnimationOption(true, "rollover", true);

}

#endif // ITEMIMAGEVECTORS_H

