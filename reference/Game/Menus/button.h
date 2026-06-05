/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: button.h, header for Button class                            #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef BUTTON_H
#define BUTTON_H

#include <memory>
#include <vector>

#include "Core/menuobject.h"
#include "Core/basicresources.h"
#include "Core/resourcemaps.h"
#include "Buildings/buildingbase.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Interfaces/ipressableobject.h"
#include "Interfaces/ilabel.h"
#include "Units/unitbase.h"


namespace Student {

/**
 * @brief The Button class is a base-class for different Tile-objects
 * in the game. \n
 *
 * Tile is responsible for:
 * * Generating resources.
 * * Checking Tile-specific object placement rules.
 * \n
 *
 * Each Tile has some Base-production which is multiplied by worker's
 * efficiency, when generating resources. Resource generation can also
 * gain flat bonuses from buildings.
 * Tiles also know how many Buildings or Workers can be placed on them.
 */
class Button : public Student::MenuObject,
               public Student::iPressableObject,
               public Student::iLabel
{
public:

    /**
     * @brief Disabled parameterless constructor.
     */
    Button() = delete;

    /**
     * @brief Constructor for the class.
     *
     */
    Button(const Course::Coordinate &coordinate,
           const int width,
           const int height,
           const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
           const std::weak_ptr<Course::iObjectManager> &objectmanager);

    Button(const std::string task,
           const Course::Coordinate& coordinate,
           const int width,
           const int height,
           const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
           const std::weak_ptr<Course::iObjectManager>& objectmanager);

    Button(const std::string task,
           const Course::Coordinate& coordinate,
           const int width,
           const int height,
           const std::string text,
           const int fontsize,
           const QColor color,
           const std::string style,
           const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
           const std::weak_ptr<Course::iObjectManager>& objectmanager);

    /**
     * @brief Default destructor.
     */
    virtual ~Button() = default;

    /**
     * @copydoc MenuObject::getType()
     */
    virtual std::string getType() const override;

    virtual void clickAction() override;

    std::string getText() override;

    void changeText(std::string text) override;

    int getFontSize() override;

    QColor getColor() override;

    std::string getStyle() override;

    void setCorrespondingTile(std::shared_ptr<Course::TileBase> tile_);

    int getMargin() override;

    void setMargin(int margin) override;

    bool noRightMargin() override;

    void setNoRightMargin(bool opt) override;

    void setHoldingIndex(int index);

    int getOffset() override;

    void setOffset(int off) override;

private:
    std::string task_;
    std::weak_ptr<Course::TileBase> correspondingTile_;

    std::string text_;
    int fontSize_;
    QColor color_;
    std::string style_;

    int margin_;
    int holdingIndex_;
    bool noRightMargin_;
    int offset_;  

}; // class Button

} // namespace Course


#endif // BUTTON_H
