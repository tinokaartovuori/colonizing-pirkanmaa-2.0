/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: iLabel.h, interface for Label                                #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef ILABEL_H
#define ILABEL_H

#include <memory>
#include <vector>
#include <QColor>

namespace Student {

class iLabel
{
public:

    virtual std::string getText() = 0;

    virtual void changeText(std::string text) = 0;

    virtual int getFontSize() = 0;

    virtual QColor getColor() = 0;

    virtual std::string getStyle() = 0;

    virtual int getMargin() = 0;

    virtual void setMargin(int margin) = 0;

    virtual bool noRightMargin() = 0;

    virtual void setNoRightMargin(bool opt) = 0;

    virtual int getOffset() = 0;

    virtual void setOffset(int off) = 0;

};

}


#endif // ILABEL_H


