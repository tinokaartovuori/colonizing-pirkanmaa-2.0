/*
######################################################################
# TIE-02402 Ohjelmointi 3: Perusteet, S2019                          #
#                                                                    #
# Project: Colonizing Pirkanmaa                                      #
# Program description: Program instructions are located in           #
#                      Documentation/documentation.pdf               #
#                                                                    #
# File: playerbase.h, header for PlayerBase-class                    #
#                                                                    #
# Authors: Otto Ranta-Ojala, 253561, otto.ranta-ojala@tuni.fi        #
#          Tino Kaartovuori, 254987, tino.kaartovuori@tuni.fi        #
######################################################################
*/


#ifndef WORLDGENERATOR_H
#define WORLDGENERATOR_H

#include <functional>
#include <map>
#include <memory>
#include <stdlib.h>

#include "Interfaces/igameeventhandler.h"
#include "Interfaces/iobjectmanager.h"
#include "Tiles/tilebase.h"
#include "Core/coordinate.h"
#include "Graphics/gamescene.h"
#include "Graphics/animationoptions.h"
#include "Graphics/animationoption.h"


namespace Student {

using TileConstructorPointer = std::function<std::shared_ptr<Course::TileBase>(
    Course::Coordinate,
    int,
    int,
    std::shared_ptr<Course::iGameEventHandler>,
    std::shared_ptr<Course::iObjectManager>)>;

/**
 * @brief The WorldGenerator class is a custom singleton world generator
 * for generating tiles for the game.
 */
class WorldGenerator
{
public:


    /**
     * @brief Used to get a reference to the Singleton instance.
     * @return Reference to the Singleton instance.
     * @post Exception guarantee: No-throw
     */
    static WorldGenerator& getInstance();

    // Prevent copy and move construction and assignment.
    WorldGenerator(const WorldGenerator&) = delete;
    WorldGenerator& operator=(const WorldGenerator&) = delete;
    WorldGenerator(WorldGenerator&&) = delete;
    WorldGenerator& operator=(WorldGenerator&&) = delete;

    /**
     * @brief Register a Tile's constructor for use in map generation.
     * @note Do this only once per Tile type or they won't be equally common.
     * Use the Tile's type as the template parameter: addConstructor<Forest>();
     * @param weight represents the rarity of the Tile, high being common.
     */
    template<typename T>
    void addConstructor(std::string name)
    {
        TileConstructorPointer ctor = std::make_shared<
                T,
                Course::Coordinate,
                int,
                int,
                std::shared_ptr<Course::iGameEventHandler>,
                std::shared_ptr<Course::iObjectManager> >;
        m_ctors.insert(std::pair<std::string, TileConstructorPointer>(name, ctor));
    }

    /**
     * @brief Generates Tile-objects and sends them to ObjectManager.
     * @param size_x is the horizontal size of the map area.
     * @param size_y is the vertical size of the map area.
     * @param seed is the seed-value used in the generation.
     * @param objectmanager points to the ObjectManager that receives the
     * generated Tiles.
     * @param eventhandler points to the student's GameEventHandler.
     * @post Exception guarantee: No-throw
     */
    void generateMap(unsigned int size_x_,
                     unsigned int size_y_,
                     unsigned int seed,
                     const std::weak_ptr<Course::iObjectManager>& objectmanager,
                     const std::weak_ptr<Course::iGameEventHandler>& eventhandler,
                     const std::weak_ptr<GameSettingsManager>& gamesettingsmanager,
                     std::weak_ptr<GameScene> scene);


    std::vector<std::vector<int>> generateTerrain(int size_x_,
                         int size_y_);

    std::tuple<std::vector<std::string>, int> getRiverOrientation(int x_, int y_, std::vector<std::vector<int>> matrix_, int num);

private:

    /**
     * @brief Default constructor.
     */
    WorldGenerator() = default;

    /**
     * @brief Default destructor.
     */
    ~WorldGenerator() = default;



    // For mapping constructors.
    std::map<std::string, TileConstructorPointer> m_ctors;

}; // class WorldGenerator

} // namespace Course


#endif // WORLDGENERATOR_H
