#include <vector>
#include <string>
#include <fstream>
#include <cmath>
#include <sstream>
#include <iomanip>
#include <iostream>
#include <filesystem>  // for std::filesystem::create_directory
#include <cmath>      // for std::sin, std::cos

struct Point {
    double x;
    double y;
};

class SVGPlotter {
private:
    // Convert coordinates to SVG viewport (handling coordinate system conversion)
    static Point transformCoord(double x, double y) {
        const double SCALE = 10;  // Scale factor to make the plot larger
        const double OFFSET = 400;  // Center point offset
        return {x * SCALE + OFFSET, -y * SCALE + OFFSET};  // Flip y-axis for SVG coordinates
    }

    static std::string colorWithOpacity(const std::string& color, double opacity) {
        if (opacity == 1.0) return color;
        return color + std::to_string(static_cast<int>(opacity * 255));
    }

public:
    static void plotMovementInteractive2D(
        const std::vector<Point>& ugvPos,
        const Point& goalPos,
        const std::vector<Point>& obstacles,
        const std::string& logDir,
        const std::string& simId,
        const std::vector<double>& obsContainer,
        const Point& etLocalMin
    ) {
        std::ofstream svg_file(logDir + "/movement_" + simId + ".svg");
        if (!svg_file.is_open()) {
            throw std::runtime_error("Could not open file for writing");
        }

        // SVG header
        svg_file << "<?xml version=\"1.0\" encoding=\"UTF-8\" standalone=\"no\"?>\n"
                 << "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"800\" height=\"800\" viewBox=\"0 0 800 800\">\n"
                 << "<style>\n"
                 << "  .scanline { stroke: orange; stroke-width: 1; stroke-dasharray: 5,5; opacity: 0.5; }\n"
                 << "  .path { stroke: blue; stroke-width: 2; fill: none; }\n"
                 << "  .point { stroke-width: 1; }\n"
                 << "</style>\n";

        // Draw grid lines (optional, for reference)
        svg_file << "<g stroke=\"#eee\" stroke-width=\"1\">\n";
        for (int i = 0; i < 800; i += 50) {
            svg_file << "<line x1=\"" << i << "\" y1=\"0\" x2=\"" << i << "\" y2=\"800\"/>\n";
            svg_file << "<line x1=\"0\" y1=\"" << i << "\" x2=\"800\" y2=\"" << i << "\"/>\n";
        }
        svg_file << "</g>\n";

        // Draw UGV path
        svg_file << "<path d=\"M";
        for (size_t i = 0; i < ugvPos.size(); ++i) {
            Point p = transformCoord(ugvPos[i].x, ugvPos[i].y);
            svg_file << p.x << " " << p.y;
            if (i < ugvPos.size() - 1) svg_file << " L ";
        }
        svg_file << "\" class=\"path\"/>\n";

        // Draw scanner traces
        for (const auto& pos : ugvPos) {
            svg_file << "<path d=\"";
            const int numPoints = 50;
            for (int i = 0; i < numPoints; ++i) {
                double theta = -M_PI/2.0 + (i * M_PI)/(numPoints-1);
                double scan_x = pos.x + 5.0 * std::cos(theta);
                double scan_y = pos.y + 5.0 * std::sin(theta);
                Point p = transformCoord(scan_x, scan_y);
                svg_file << (i == 0 ? "M" : "L") << p.x << " " << p.y << " ";
            }
            svg_file << "\" class=\"scanline\"/>\n";
        }

        // Draw obstacles
        for (const auto& obs : obstacles) {
            Point p = transformCoord(obs.x, obs.y);
            svg_file << "<circle cx=\"" << p.x << "\" cy=\"" << p.y 
                    << "\" r=\"3\" class=\"point\" fill=\"red\"/>\n";
        }

        // Draw obstacle container
        Point p1 = transformCoord(obsContainer[0], obsContainer[1]);
        Point p2 = transformCoord(obsContainer[2], obsContainer[3]);
        
        // Calculate container corners (similar to original code)
        double dx = obsContainer[2] - obsContainer[0];
        double dy = obsContainer[3] - obsContainer[1];
        double length = std::sqrt(dx*dx + dy*dy);
        double nx = -dy/length * 6.25;  // Half width = 12.5/2
        double ny = dx/length * 6.25;
        
        std::vector<Point> corners = {
            transformCoord(obsContainer[0] + nx, obsContainer[1] + ny),
            transformCoord(obsContainer[0] - nx, obsContainer[1] - ny),
            transformCoord(obsContainer[2] - nx, obsContainer[3] - ny),
            transformCoord(obsContainer[2] + nx, obsContainer[3] + ny)
        };

        // Draw container area
        svg_file << "<path d=\"M";
        for (size_t i = 0; i < corners.size(); ++i) {
            svg_file << corners[i].x << " " << corners[i].y;
            if (i < corners.size() - 1) svg_file << " L ";
        }
        svg_file << " Z\" fill=\"rgba(255,0,0,0.2)\" stroke=\"purple\"/>\n";

        // Draw start, goal, and local minimum points
        Point start = transformCoord(ugvPos[0].x, ugvPos[0].y);
        Point goal = transformCoord(goalPos.x, goalPos.y);
        Point localMin = transformCoord(etLocalMin.x, etLocalMin.y);

        svg_file << "<circle cx=\"" << start.x << "\" cy=\"" << start.y 
                << "\" r=\"5\" class=\"point\" fill=\"green\"/>\n"
                << "<circle cx=\"" << goal.x << "\" cy=\"" << goal.y 
                << "\" r=\"5\" class=\"point\" fill=\"black\"/>\n"
                << "<circle cx=\"" << localMin.x << "\" cy=\"" << localMin.y 
                << "\" r=\"5\" class=\"point\" fill=\"black\"/>\n";

        // Add legend
        svg_file << "<g transform=\"translate(650,50)\" font-family=\"Arial\" font-size=\"12\">\n"
                << "<text y=\"20\">Legend:</text>\n"
                << "<circle cx=\"10\" cy=\"40\" r=\"3\" fill=\"green\"/><text x=\"20\" y=\"45\">Start</text>\n"
                << "<circle cx=\"10\" cy=\"60\" r=\"3\" fill=\"black\"/><text x=\"20\" y=\"65\">Goal</text>\n"
                << "<circle cx=\"10\" cy=\"80\" r=\"3\" fill=\"red\"/><text x=\"20\" y=\"85\">Obstacles</text>\n"
                << "<line x1=\"10\" y1=\"100\" x2=\"30\" y2=\"100\" class=\"scanline\"/>"
                << "<text x=\"35\" y=\"105\">Scanner</text>\n"
                << "</g>\n";

        // Close SVG
        svg_file << "</svg>";
        svg_file.close();
    }
};

int main() {
    try {
        // Create a test directory if it doesn't exist
        std::string logDir = "./plot_output";
        std::filesystem::create_directory(logDir);

        // Generate some sample UGV positions (a spiral path)
        std::vector<Point> ugvPositions;
        for (double t = 0; t < 10; t += 0.1) {
            double r = t / 2;  // Increasing radius
            ugvPositions.push_back({
                r * std::cos(t),
                r * std::sin(t)
            });
        }

        // Set a goal position
        Point goalPosition = {8, 8};

        // Generate some random obstacles
        std::vector<Point> obstacles;
        for (int i = 0; i < 20; i++) {
            // Use a simple pseudo-random distribution
            double x = std::sin(i * 1.7) * 10;
            double y = std::cos(i * 2.3) * 10;
            obstacles.push_back({x, y});
        }

        // Define obstacle container (a simple line segment)
        std::vector<double> obstacleContainer = {2, 2, 6, 6};  // From (2,2) to (6,6)

        // Set a local minimum point
        Point localMin = {4, 4};

        // Create the plot
        std::cout << "Creating plot..." << std::endl;
        SVGPlotter::plotMovementInteractive2D(
            ugvPositions,
            goalPosition,
            obstacles,
            logDir,
            "test_plot",
            obstacleContainer,
            localMin
        );

        std::cout << "Plot created successfully! Check " << logDir << "/movement_test_plot.svg" << std::endl;
        
        // Print some statistics
        std::cout << "\nPlot Statistics:" << std::endl;
        std::cout << "Number of UGV positions: " << ugvPositions.size() << std::endl;
        std::cout << "Number of obstacles: " << obstacles.size() << std::endl;
        std::cout << "Goal position: (" << goalPosition.x << ", " << goalPosition.y << ")" << std::endl;
        std::cout << "Local minimum: (" << localMin.x << ", " << localMin.y << ")" << std::endl;

        return 0;

    } catch (const std::exception& e) {
        std::cerr << "Error: " << e.what() << std::endl;
        return 1;
    }
}