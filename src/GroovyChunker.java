import org.codehaus.groovy.ast.*;
import org.codehaus.groovy.ast.stmt.Statement;
import org.codehaus.groovy.control.*;

import java.io.IOException;
import java.lang.reflect.Method;
import java.nio.file.Files;
import java.nio.file.Path;
import java.nio.file.Paths;
import java.util.*;

/**
 * GroovyChunker – splits Groovy source into chunks suitable for RAG pipelines.
 *   • One header chunk per class / interface / trait (including nested ones)
 *   • One chunk per method
 *   • One chunk per constructor
 *
 * Header chunk includes: package, imports, Groovydoc, declaration line(s),
 *                        fields, object & static initialiser blocks.
 *
 * Runtime jars needed:
 *   org.codehaus.groovy:groovy:3.0.21          (or any 3.x you use)
 *   com.tunnelvisionlabs:antlr4-runtime:4.9.0  (Groovy’s tiny ANTLR helper)
 *
 * Adapt {@link #processChunkForEmbedding} to feed chunks into your embedding store.
 */
public class GroovyChunker {

    public static void main(String[] args) throws IOException {
        if (args.length == 0) {
            System.out.println("Usage: java GroovyChunker <file.groovy>");
            return;
        }
        groovyChunker(args[0]);
    }

    /* ────────────────────────────────────────────────────────── */

    public static void groovyChunker(String fileName) throws IOException {

        Path srcPath          = Paths.get(fileName);
        List<String> allLines = Files.readAllLines(srcPath);
        String sourceText     = String.join("\n", allLines);

        CompilerConfiguration cfg = new CompilerConfiguration();
        SourceUnit su = new SourceUnit(
                srcPath.toString(), sourceText, cfg, null, new ErrorCollector(cfg));
        su.parse();
        su.completePhase();
        su.convert();

        ModuleNode module = su.getAST();

        String packageLine = module.getPackage() == null ? ""
                : "package " + module.getPackageName() + ";\n\n";

        List<String> importLines = new ArrayList<>();
        module.getImports()              .forEach(i -> importLines.add(i.getText() + "\n"));
        module.getStarImports()          .forEach(i -> importLines.add(i.getText() + "\n"));
        module.getStaticImports().values()
                .forEach(i -> importLines.add(i.getText() + "\n"));
        module.getStaticStarImports().values()
                .forEach(i -> importLines.add(i.getText() + "\n"));
        if (!importLines.isEmpty()) importLines.add("\n");

        for (ClassNode top : module.getClasses()) {
            walkClass(top, packageLine, importLines, allLines);
        }
    }

    /* depth-first walk through nested / inner classes */
    private static void walkClass(ClassNode clazz,
                                  String packageLine,
                                  List<String> importLines,
                                  List<String> allLines) {

        processChunkForEmbedding(
                buildHeaderChunk(clazz, packageLine, importLines, allLines));

        for (MethodNode m : clazz.getMethods()) {
            if (m.isSynthetic()) continue;
            processChunkForEmbedding(
                    buildMemberChunk(clazz, m, packageLine, allLines));
        }
        for (MethodNode ctor : clazz.getDeclaredConstructors()) {
            if (ctor.isSynthetic()) continue;
            processChunkForEmbedding(
                    buildMemberChunk(clazz, ctor, packageLine, allLines));
        }

        Iterator<InnerClassNode> it = clazz.getInnerClasses();
        while (it.hasNext()) {
            InnerClassNode inner = it.next();
            if (inner.getNameWithoutPackage() != null) {
                walkClass(inner, packageLine, importLines, allLines);
            }
        }
    }

    /* ───── helpers ─────────────────────────────────────────── */

    private static String buildHeaderChunk(ClassNode clazz,
                                           String packageLine,
                                           List<String> importLines,
                                           List<String> lines) {

        StringBuilder sb = new StringBuilder();
        sb.append(packageLine);
        importLines.forEach(sb::append);

        /* Groovydoc immediately above the class? */
        int classStartL = clazz.getLineNumber();
        int ptr = classStartL - 2;
        if (ptr >= 0 && lines.get(ptr).trim().endsWith("*/")) {
            while (ptr >= 0 && !lines.get(ptr).contains("/**")) ptr--;
            if (ptr >= 0)
                for (int i = ptr; i < classStartL - 1; i++)
                    sb.append(lines.get(i)).append('\n');
        }

        /* declaration line(s) up to first '{' */
        int openBraceLine = classStartL - 1;
        while (openBraceLine < lines.size() && !lines.get(openBraceLine).contains("{"))
            openBraceLine++;
        for (int i = classStartL - 1; i <= openBraceLine; i++)
            sb.append(lines.get(i)).append('\n');
        sb.append('\n');

        /* fields */
        clazz.getFields().stream()
                .filter(f -> !f.isSynthetic())
                .forEach(f -> sb.append(slice(lines, f.getLineNumber(),
                        f.getLastLineNumber())).append('\n'));

        /* object initialisers */
        sliceStatements(clazz.getObjectInitializerStatements(), lines, sb);

        /* static initialisers – obtained reflectively for maximum compatibility */
        sliceStatements(getStaticInitStatements(clazz), lines, sb);

        sb.append("}\n");
        return sb.toString();
    }

    private static String buildMemberChunk(ClassNode clazz,
                                           MethodNode m,
                                           String packageLine,
                                           List<String> lines) {

        StringBuilder sb = new StringBuilder();
        sb.append(packageLine);
        sb.append("class ").append(clazz.getNameWithoutPackage()).append(" {\n\n");
        sb.append(slice(lines, m.getLineNumber(), m.getLastLineNumber())).append('\n');
        sb.append("}\n");
        return sb.toString();
    }

    /* reflection helper: returns static-init statements if the JVM method exists */
    @SuppressWarnings("unchecked")
    private static List<Statement> getStaticInitStatements(ClassNode clazz) {
        try {
            Method m = ClassNode.class.getMethod("getStaticClassInitStatements");
            return (List<Statement>) m.invoke(clazz);
        } catch (NoSuchMethodException e) {            // method name differs
            try {
                Method m = ClassNode.class.getMethod("getStaticInitializerStatements");
                return (List<Statement>) m.invoke(clazz);
            } catch (Exception ignored) {
                return Collections.emptyList();
            }
        } catch (Exception e) {
            return Collections.emptyList();
        }
    }

    /* slice original source lines[start..end] (inclusive, 1-based) */
    private static String slice(List<String> lines, int startL, int endL) {
        if (startL <= 0 || endL <= 0) return "";
        StringBuilder out = new StringBuilder();
        for (int i = startL - 1; i <= endL - 1 && i < lines.size(); i++) {
            out.append(lines.get(i)).append('\n');
        }
        return out.toString();
    }

    private static void sliceStatements(List<Statement> stmts,
                                        List<String> lines,
                                        StringBuilder out) {
        for (Statement st : stmts) {
            out.append(slice(lines, st.getLineNumber(), st.getLastLineNumber()))
                    .append('\n');
        }
    }

    /* Replace with your embedding / storage logic. */
    private static void processChunkForEmbedding(String chunk) {
        System.out.println("=== EMBEDDING CHUNK START ===");
        System.out.println(chunk);
        System.out.println("=== EMBEDDING CHUNK END ===\n");
    }
}
