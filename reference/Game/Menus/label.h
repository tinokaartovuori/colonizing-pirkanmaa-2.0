/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: label.h, header for Label class                              #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef LABEL_H
#define LABEL_H

#include <memory>
#include <vector>

#include <QtGlobal> // For Q_ASSERT
#include <QDebug>
#include <QColor>

#include "Core/menuobject.h"
#include "Core/basicresources.h"
#include "Core/resourcemaps.h"

#include "Buildings/buildingbase.h"
#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Interfaces/ilabel.h"
#include "Units/unitbase.h"


namespace Student {

/**
 * @brief The Label class is a base-class for different Tile-objects
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
class Label : public Student::MenuObject, public Student::iLabel
{
public:

    /**
     * @brief Disabled parameterless constructor.
     */
    Label() = delete;

    /**
     * @brief Constructor for the class.
     *
     */
    Label(const Course::Coordinate& coordinate,
           const int width,
           const int height, const std::string text,
          const int fontsize,
          const QColor color,
          const std::string style,
           const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
           const std::weak_ptr<Course::iObjectManager> &objectmanager);

    /**
     * @brief Default destructor.
     */
    virtual ~Label() = default;

    /**
     * @copydoc MenuObject::getType()
     */
    virtual std::string getType() const override;

    std::string getText() override;

    void changeText(std::string text) override;

    int getFontSize() override;

    QColor getColor() override;

    std::string getStyle() override;

    int getMargin() override;

    void setMargin(int margin) override;

    int getOffset() override;

    void setOffset(int off) override;

    bool noRightMargin() override;

    void setNoRightMargin(bool opt) override;

private:

    std::string text_;
    int fontSize_;
    QColor color_;
    std::string style_;
    int offset_;

    int margin_;
    bool noRightMargin_;

}; // class Label

} // namespace Course


#endif // LABEL_H
