/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: label.cpp                                                    #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "label.h"

#include "Exceptions/notenoughspace.h"
#include "Exceptions/ownerconflict.h"
#include "Exceptions/invalidpointer.h"
#include "Core/playerbase.h"


namespace Student {

Label::Label(const Course::Coordinate& coordinate,
             const int width,
             const int height,
             const std::string text,
             const int fontsize,
             const QColor color,
             const std::string style,
             const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
             const std::weak_ptr<Course::iObjectManager>& objectmanager):
             MenuObject(coordinate, width, height, eventhandler, objectmanager),
             text_(text),
             fontSize_(fontsize),
             color_(color),
             style_(style)
{
    margin_ = 6;
    noRightMargin_ = false;
    offset_ = 0;
}

std::string Label::getType() const
{
    return "Label";
}

std::string Label::getText()
{
    return text_;
}

void Label::changeText(std::string text) {
    text_ = text;
}

int Label::getFontSize()
{
    return fontSize_;
}

QColor Label::getColor()
{
    return color_;
}

std::string Label::getStyle()
{
    return style_;
}

int Label::getMargin()
{
    return margin_;
}

void Label::setMargin(int margin)
{
    margin_ = margin;
}

int Label::getOffset()
{
    return offset_;
}

void Label::setOffset(int off)
{
    offset_ = off;
}

bool Label::noRightMargin() {
    return noRightMargin_;
}

void Label::setNoRightMargin(bool opt) {
    noRightMargin_ = opt;
}

} // namespace Course
