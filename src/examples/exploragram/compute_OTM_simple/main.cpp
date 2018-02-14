#include <geogram/basic/common.h>
#include <geogram/basic/logger.h>
#include <geogram/basic/command_line.h>
#include <geogram/basic/command_line_args.h>
#include <geogram/basic/stopwatch.h>
#include <geogram/basic/file_system.h>
#include <geogram/basic/process.h>
#include <geogram/basic/progress.h>
#include <geogram/mesh/mesh.h>
#include <geogram/mesh/mesh_io.h>
#include <geogram/mesh/mesh_tetrahedralize.h>
#include <geogram/voronoi/CVT.h>

/* #include <exploragram/optimal_transport/optimal_transport_on_surface.h> */
#include <exploragram/optimal_transport/optimal_transport_3d.h>
#include <exploragram/optimal_transport/sampling.h>
#include <geogram/NL/nl.h>
#include <geogram/NL/nl_matrix.h>

namespace {
    using namespace GEO;

    /**
     * \brief Loads a volumetric mesh.
     * \details If the specified file contains a surface, try to
     *  tesselate it. If the surface has self-intersections, try to
     *  remove them.
     * \param[in] filename the name of the file
     * \param[out] M the mesh
     * \retval true if the file was successfully loaded
     * \retval false otherwise
     */
    bool load_volume_mesh(const std::string& filename, Mesh& M) {
        MeshIOFlags flags;
        flags.set_element(MESH_CELLS);
        flags.set_attribute(MESH_CELL_REGION);

        if(!mesh_load(filename, M, flags)) {
            return 1;
        }
        if(!M.cells.are_simplices()) {
            Logger::err("I/O") << "File "
                << filename
                << " should only have tetrahedra" << std::endl;
            return false;
        }
        if(M.cells.nb() == 0) {
            Logger::out("I/O") << "File "
                << filename
                << " does not contain a volume" << std::endl;
            Logger::out("I/O") << "Trying to tetrahedralize..." << std::endl;
            if(!mesh_tetrahedralize(M,true,false)) {
                return false;
            }
        }
        return true;
    }
}

int main(int argc, char** argv) {
    using namespace GEO;

    GEO::initialize();

    try {

        std::vector<std::string> filenames;

        CmdLine::import_arg_group("standard");
        CmdLine::import_arg_group("algo");
        CmdLine::import_arg_group("opt");
        CmdLine::declare_arg("nb_iter", 1000, "number of iterations for OTM");
        CmdLine::declare_arg("RDT", false, "save regular triangulation");
        CmdLine::declare_arg_group(
                                   "RVD", "RVD output options", CmdLine::ARG_ADVANCED
                                  );
        CmdLine::declare_arg("RVD", false, "save restricted Voronoi diagram");

        CmdLine::declare_arg(
                             "epsilon", 0.01, "relative measure error in a cell"
                            );
        CmdLine::set_arg("algo:delaunay", "BPOW");
        CmdLine::declare_arg(
                             "density_min", 1.0, "min density in first mesh"
                            );
        CmdLine::declare_arg(
                             "density_max", 1.0, "max density in first mesh"
                            );
        CmdLine::declare_arg(
                             "density_function", "x", "used function for density"
                            );

        Logger::div("Warpdrive - Optimal Transport");

        if( !CmdLine::parse(argc, argv, filenames, "mesh1 points")) {
            return 1;
        }

        std::string mesh1_filename = filenames[0];
        std::string points_filename = filenames[1];

        Logger::div("Loading data");

        Mesh M1;
        Mesh M2_samples;

        // Source mesh
        if(!load_volume_mesh(mesh1_filename, M1)) {
            return 1;
        }

        // Source density
        set_density(
                    M1,
                    CmdLine::get_arg_double("density_min"),
                    CmdLine::get_arg_double("density_max"),
                    CmdLine::get_arg("density_function")
                   );

        if(M1.cells.nb() == 0) {
            Logger::err("Mesh") << "M1 does not have any tetrahedron, exiting"
                << std::endl;
            return 1;
        }

        // Target point cloud
        if(!mesh_load(points_filename, M2_samples)) {
            return 1;
        }

        Logger::div("Optimal transport");
        // Everything happens in dimension 4 (power diagram is seen
        // as Voronoi diagram in dimension 4), therefore the dimension
        // of M1 needs to be changed as well (even if it is not used).
        M1.vertices.set_dimension(4);

        OptimalTransportMap3d OTM(&M1);
        OTM.set_epsilon(CmdLine::get_arg_double("epsilon"));
        index_t nb_iter = CmdLine::get_arg_uint("nb_iter");

        OTM.set_points(M2_samples.vertices.nb(), M2_samples.vertices.point_ptr(0));
	    OTM.set_Newton(true);
        OTM.set_verbose(false);
        // Target density
        index_t N = OTM.nb_points();
        for (index_t i = 0; i < N; ++i) {
            OTM.set_nu(i, 1.0 * OTM.total_mesh_mass() / N);
        }

        /* for (index_t i = 0; i < N; ++i) { */
        /*     Logger::out("") << OTM.nu(i) << " "; */
        /* } */

        std::vector<double> weights(N);
        for (index_t i = 0; i < N; ++i) {
            weights[i] = 0.0;
        }
        std::vector<double> grad(N);
        std::vector<double> pk(N);
        double f = 0.0;
        OTM.new_linear_system(N,pk.data());
        OTM.eval_func_grad_Hessian(N, weights.data(), f, grad.data());

        NLSparseMatrix* M = nlGetCurrentSparseMatrix();

        // rows
        for (NLuint i = 0; i < M->m; ++i) {
            NLRowColumn row = M->row[i];
            for (NLuint c = 0; c < row.size; ++c) {
                NLCoeff coeff = row.coeff[c];
                std::cout << "(i, j) = " << i << ", " << coeff.index << " = " << coeff.value << std::endl;
            }
        }

        // diag
        for (NLuint i = 0; i < M->diag_size; ++i) {
            std::cout << "(i, i) = " << i << ", " << i << " = " << M->diag[i] << std::endl;
        }


        Logger::out("") << "f = " << f << std::endl;
        for (index_t i = 0; i < N; ++i) {
            std::cout << grad[i] + OTM.nu(i) << " ";
        }

        {
            Stopwatch W("OTM Total");
            OTM.optimize_full_Newton(nb_iter);
        }

        for (index_t i = 0; i < N; ++i) {
            weights[i] = OTM.weight(i);
            Logger::out("") << OTM.weight(i) << " ";
        }
    }
    catch(const std::exception& e) {
        std::cerr << "Received an exception: " << e.what() << std::endl;
        return 1;
    }

    Logger::out("") << "Everything OK, Returning status 0" << std::endl;
    return 0;
}

