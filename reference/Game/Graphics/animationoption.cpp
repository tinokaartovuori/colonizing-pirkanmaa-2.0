/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: animationoption.cpp, see animationoption.h for more info     #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "animationoption.h"

namespace Student {

AnimationOption::AnimationOption()
{
    m_animated = false;
    m_style = "rollover";
    m_randomFrame = false;
}

AnimationOption::AnimationOption(bool onoff):
    m_animated(onoff)
{
    m_style = "rollover";
    m_randomFrame = false;
}

AnimationOption::AnimationOption(bool onoff, std::string style):
    m_animated(onoff), m_style(style)
{
    m_randomFrame = false;
}

AnimationOption::AnimationOption(bool onoff, std::string style, bool randomFrame):
    m_animated(onoff), m_style(style), m_randomFrame(randomFrame)
{
}

bool AnimationOption::isAnimated() const
{
    return m_animated;
}

bool AnimationOption::startRandomFrame() const
{
    return m_randomFrame;
}

std::string AnimationOption::getStyle() const
{
    return m_style;
}


} // namespace Student
