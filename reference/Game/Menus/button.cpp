/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: button.cpp                                                   #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#include "button.h"

#include "helpwindow.h"

#include <QtGlobal> // For Q_ASSERT
#include <QDebug>

#include "Exceptions/notenoughspace.h"
#include "Exceptions/ownerconflict.h"
#include "Exceptions/invalidpointer.h"
#include "Core/playerbase.h"


namespace Student {

Button::Button(
        const Course::Coordinate& coordinate,
                   const int width,
                   const int height,
                   const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                   const std::weak_ptr<Course::iObjectManager>& objectmanager):
    MenuObject(coordinate, width, height, eventhandler, objectmanager)
{
    task_ = "none";
    text_ = "",
    fontSize_ = 1,
    color_ = QColor(0, 0, 0),
    style_ = "CENTER";
    margin_ = 6;
}

Button::Button(
        const std::string task,
        const Course::Coordinate& coordinate,
                   const int width,
                   const int height,
                   const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
                   const std::weak_ptr<Course::iObjectManager>& objectmanager):
    MenuObject(coordinate, width, height, eventhandler, objectmanager),
    task_(task)
{
    text_ = "",
    fontSize_ = 1,
    color_ = QColor(0, 0, 0),
    style_ = "CENTER";
    margin_ = 6;
    holdingIndex_ = 0;
}

Button::Button(const std::string task,
               const Course::Coordinate &coordinate,
               const int width,
               const int height,
               const std::string text,
               const int fontsize,
               const QColor color,
               const std::string style,
               const std::weak_ptr<Course::iGameEventHandler>
               &eventhandler, const std::weak_ptr<Course::iObjectManager>
               &objectmanager):
    MenuObject(coordinate, width, height, eventhandler, objectmanager),
    task_(task),
    text_(text),
    fontSize_(fontsize),
    color_(color),
    style_(style)
{
    margin_ = 6;
    holdingIndex_ = 0;
    offset_ = 0;
    noRightMargin_ = false;
}

std::string Button::getType() const
{
    return "Button";
}

void Button::clickAction()
{

    if (task_ == "opendefaultmenu") {
        lockEventHandler()->openDefaultMenuView();
    }
    else if (task_.find("addUnit") != std::string::npos) {
        if (task_.find("BasicWorker") != std::string::npos) {
            lockEventHandler()->createUnit("BasicWorker");
        }
        else if (task_.find("Expert") != std::string::npos) {
            lockEventHandler()->createUnit("Expert");
        }
        else if (task_.find("Soldier") != std::string::npos) {
            lockEventHandler()->createUnit("Soldier");
        }
    }
    else if (task_ == "openbuymenu") {
        lockEventHandler()->openUnitBuyMenu();
    }
    else if (task_ == "openstatsmenu") {
        lockEventHandler()->openStatsMenuView();
    }
    else if (task_ == "endturn") {
        lockEventHandler()->endTurn();
    }
    else if (task_ == "help") {
        /*HelpWindow* helpWindow = new HelpWindow();
        helpWindow->setAttribute(Qt::WA_DeleteOnClose);
        helpWindow->show();*/
    }

    else if (task_.find("moveUnit") != std::string::npos and
             correspondingTile_.lock() != nullptr)
    {
        if (task_.find("0") != std::string::npos) {
            lockEventHandler()->moveUnitFromTile(0, correspondingTile_.lock());
        }
        else if (task_.find("1") != std::string::npos) {
            lockEventHandler()->moveUnitFromTile(1, correspondingTile_.lock());
        }
        else if (task_.find("2") != std::string::npos) {
            lockEventHandler()->moveUnitFromTile(2, correspondingTile_.lock());
       }

    }
    else if (task_.find("delUnit") != std::string::npos) {
        if (task_.find("0") != std::string::npos) {
            lockEventHandler()->deleteUnitFromTile(0, correspondingTile_.lock());
        }
        else if (task_.find("1") != std::string::npos) {
            lockEventHandler()->deleteUnitFromTile(1, correspondingTile_.lock());
        }
        else if (task_.find("2") != std::string::npos) {
            lockEventHandler()->deleteUnitFromTile(2, correspondingTile_.lock());
        }
    }

    else if (task_.find("build(") != std::string::npos)
    {
        if (task_.find("Village") != std::string::npos)
        {
            lockEventHandler()->buildBuilding("Village",
                                              correspondingTile_.lock());
        }
        else if (task_.find("Outpost") != std::string::npos)
        {
            lockEventHandler()->buildBuilding("Outpost",
                                              correspondingTile_.lock());
        }
        else if (task_.find("Nuclear Power Plant") != std::string::npos) {
            lockEventHandler()->buildBuilding("Nuclear Power Plant",
                                         correspondingTile_.lock());
        }
        else if (task_.find("Mine") != std::string::npos) {
            lockEventHandler()->buildBuilding("Mine", correspondingTile_.lock());
        }
        else if (task_.find("Hydroelectric Power Plant") != std::string::npos) {
            lockEventHandler()->buildBuilding("Hydroelectric Power Plant",
                                              correspondingTile_.lock());
        }
        else if (task_.find("Farm") != std::string::npos) {
            lockEventHandler()->buildBuilding("Farm", correspondingTile_.lock());
        }
        else if (task_.find("Build") != std::string::npos) {
            lockEventHandler()->buildBuilding("Build", correspondingTile_.lock());
        }
        else if (task_.find("Bridge") != std::string::npos) {
            lockEventHandler()->buildBuilding("Bridge", correspondingTile_.lock());
        }

    }
    else if (task_.find("switchBuyMenu") != std::string::npos) {
        lockEventHandler()->setTileInspectionMenuView
                   (correspondingTile_.lock(), holdingIndex_);
    }
    else if (task_.find("newGame") != std::string::npos) {
        lockEventHandler()->restartGame();
    }
    else if (task_.find("quit") != std::string::npos) {
        lockObjectManager()->getGameScene()->deleteObjects();
        exit(0);
    }
    else {
        qDebug() << "No task for button";
    }
}

int Button::getOffset()
{
    return offset_;
}

void Button::setOffset(int off)
{
    offset_ = off;
}

std::string Button::getText()
{
    return text_;
}

void Button::changeText(std::string text)
{
    text_ = text;
}

int Button::getFontSize()
{
    return fontSize_;
}

QColor Button::getColor()
{
    return color_;
}

std::string Button::getStyle()
{
    return style_;
}

int Button::getMargin()
{
    return margin_;
}

void Button::setMargin(int margin)
{
    margin_ = margin;
}

void Button::setHoldingIndex(int index)
{
    holdingIndex_ = index;
}

bool Button::noRightMargin() {
    return noRightMargin_;
}

void Button::setNoRightMargin(bool opt) {
    noRightMargin_ = opt;
}


void Button::setCorrespondingTile(std::shared_ptr<Course::TileBase> tile_)
{
    correspondingTile_ = tile_;
}


} // namespace Course
