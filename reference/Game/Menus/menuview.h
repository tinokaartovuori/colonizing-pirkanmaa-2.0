/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: menuview.h, header for MenuView class                        #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/

#ifndef MENUVIEW_H
#define MENUVIEW_H

#include <memory>
#include <vector>

#include "Core/basicresources.h"
#include "Core/resourcemaps.h"
#include "Buildings/buildingbase.h"
#include "Units/unitbase.h"

#include "Graphics/animationoptions.h"


#include "Interfaces/icontainer.h"


namespace Student {

/**
 * @brief The MenuView class is a base-class for different Tile-objects
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
class MenuView : public Student::MenuObject, public Student::iContainer
{
public:

    /**
     * @brief Disabled parameterless constructor.
     */
    MenuView() = delete;

    /**
     * @brief Constructor for the class.
     *
     */

    MenuView(const Course::Coordinate &coordinate,
             int width,
             int height,
             int gridSize,
             const std::weak_ptr<Course::iGameEventHandler> &eventhandler,
             const std::weak_ptr<Course::iObjectManager> &objectmanager);


    MenuView(const Course::Coordinate& coordinate,
             const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
             const std::weak_ptr<Course::iObjectManager>& objectmanager
             );

    /**
     * @brief Default destructor.
     */
    virtual ~MenuView() = default;

    /**
     * @copydoc MenuObject::getType()
     */
    virtual std::string getType() const override;

    void addMenuObject(const std::shared_ptr<MenuObject>& obj) override;

    std::vector<std::shared_ptr<MenuObject>> getMenuObjects() const override;

    int getGridSize() const;

    QPoint getAbsoluteAdder();


private:

    std::vector<std::shared_ptr<MenuObject>> m_upperLayer;
    int m_gridSize;


}; // class MenuView

} // namespace Course


#endif // MENUVIEW_H
