/*
 * Author: Blake McBride
 * Date: 2/16/20
 *
 * I've found that I sometimes spend more time messing with build programs (such 
 * as Maven, Gradle, and others) than the underlying application I am trying to 
 * build.  They all do the normal things very, very easily.  But when you try to
 * go off their beaten path it gets real difficult real fast.  Being sick and
 * tired of this, and having easily built a shell script to build what I want, I
 * needed a more portable solution.  The files in this directory are that solution.
 *
 * It should be noted, however, that unlike a shell script, this build system 
 * does not execute commands that are already done.  In other words, only the 
 * minimum steps necessary to rebuild a system are actually executed.  So, this 
 * build system runs as fast as the others.
 *
 * There are two classes as follows:
 *
 *     BuildUtils -  the generic utilities needed to build
 *     Tasks      -  the application-specific build procedures (or tasks)
 *
 *    Non-private instance methods with no parameters are considered tasks.
 */

package builder;

import java.io.File;

import static builder.BuildUtils.*;


/**
 * This class contains the tasks that are executed by the build system.
 * <br><br>
 * The build system finds the names of the tasks through reflection.
 * It also does camelCase conversion.  So a task named abcDef may be evoked
 * as abc-def.
 * <br><br>
 * Each task must be declared as a public static method with no parameters.
 */
public class Tasks {

    final static String LIBS = "libs";
    final static ForeignDependencies foreignLibs = buildForeignDependencies();
    final static LocalDependencies localLibs = buildLocalDependencies();
    final static String BUILDDIR = "target";
    private static String [] args;

    /**
     * Main entry point for the build system.  It tells the build system what arguments were passed in
     * and what class contains all the tasks.
     *
     * @param args the arguments to the program
     * @throws Exception if exception is thrown
     * @throws InstantiationException if the class cannot be instantiated
     */
    public static void main(String[] args) throws Exception {
        Tasks.args = args;
        BuildUtils.build(args, Tasks.class, LIBS);
    }

    /**
     * Display a list of valid tasks.  It is called by the build system
     * when the user selects the 'list-tasks' task.
     * <br><br>
     * The build system expects this method to be named listTasks.
     *
     * @see BuildUtils#build
     */
    public static void listTasks() {
        println("");
        println("build                    build the entire system but don't run it");
        println("");

        println("Build the entire system and run it:");
        println("  run <full-path-of-class-to-run>  [argument]...                     ");
        println("Example:  bld run org.example.Main  the-first-argument  the-second-argument");

        println("");
        println("clean                    remove all compiled files");
        println("realclean                + remove downloaded jar files");
        println("ideclean                 + IDE files");
        println("");

        println("jar                      build application jar file");
        println("javadoc                  build javadoc files");
        println("");

        println("libs                     download foreign jar files");
        println("");
    }

    /**
     * Build the system
     */
    public static void build() {
        libs();
        //mkdir(BUILDDIR + "/classes");
        buildJava("src", "src", localLibs, foreignLibs, null);
    }

    /**
     * Build the system and run it
     */
    public static void run() {
        build();
        runJava("src", "GroovyChunker", localLibs, foreignLibs);
    }

    /**
     * Create Kiss.jar.  This is a JAR file that can be used in other apps as a
     * utility library.
     */
    public static void jar() {
        build();
        createJar(BUILDDIR + "/classes", "app.jar");
    }

    /**
     * Download needed foreign libraries
     */
    public static void libs() {
        downloadAll(foreignLibs);
    }

    /**
     * build the javdoc files
     */
    public static void javadoc() {
        libs();
        buildJavadoc("src/main/java", LIBS, BUILDDIR + "/javadoc", "JavaDocOverview.html");
    }

    /**
     * Remove:<br>
     * -- all files that were built<br><br>
     * Do not remove:<br>
     * -- the IDE files
     */
    public static void clean() {
        rmRegex("src", ".*\\.class");
    }

    /**
     * Remove:<br>
     * -- all files that were built<br>
     * Do not remove:<br>
     * -- the IDE files
     */
    public static void realclean() {
        clean();
        rmRegex("builder", ".*\\.class");
        delete(foreignLibs);
    }

    /**
     * Remove:<br>
     * -- all files that were built<br>
     * -- the IDE files
     */
    public static void ideclean() {
        realclean();

        rmTree(".project");
        rmTree(".settings");
        rmTree(".vscode");

        // IntelliJ
        rmTree(".idea");
        rmTree("out");
        rmRegex(".", ".*\\.iml");
        rmRegex("src", ".*\\.iml");

        // NetBeans
        rmTree("dist");
        rmTree("nbproject");
        rmTree("build");
        rm("nbbuild.xml");
    }

    /**
     * Specify the jars used by the system but not included in the distribution.
     * These are the jars that are to be downloaded by the build system.
     *
     * @return
     */
    private static ForeignDependencies buildForeignDependencies() {
        final ForeignDependencies dep = new ForeignDependencies();
        // Groovy core (provides the AST classes used by GroovyChunker)
        dep.add(LIBS,
    "https://repo1.maven.org/maven2/org/codehaus/groovy/groovy/3.0.21/groovy-3.0.21.jar");

        // Groovy 3’s new Parrot parser relies on ANTLR-4; this ~430 KB jar satisfies it
        dep.add(LIBS,
    "https://repo1.maven.org/maven2/com/tunnelvisionlabs/antlr4-runtime/4.9.0/antlr4-runtime-4.9.0.jar");

        return dep;
    }

    /**
     * This specifies the jar files used by the system that are included in the distribution.
     * (All are open-source but exist in other projects.)
     *
     * @return
     */
    private static LocalDependencies buildLocalDependencies() {
        final LocalDependencies dep = new LocalDependencies();
        //dep.add(LIBS + "/javaparser-core-3.26.4.jar");
        return dep;
    }

}
