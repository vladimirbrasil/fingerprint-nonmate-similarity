import com.machinezoo.sourceafis.*;
import java.io.*;
import java.nio.*;
import java.nio.file.*;
import java.util.*;
import java.util.stream.*;

// Todos-contra-todos com SourceAFIS, na mesma ordem de out/index.csv.
// Saída: matriz NxN float32 little-endian (diagonal = auto-comparação) em out/sourceafis/matrix.f32.
// Templates ficam em cache (out/sourceafis/templates/) — rodar de novo é idempotente.
public class AllVsAll {
    // Resolução nominal de cada base (documentação oficial das FVC).
    static double dpi(String base) {
        if (base.equals("FVC2002_DB2_B")) return 569;
        if (base.equals("FVC2004_DB3_B")) return 512;
        return 500;
    }

    public static void main(String[] args) throws Exception {
        Path root = Paths.get(args.length > 0 ? args[0] : ".").toAbsolutePath();
        Path out = root.resolve("out/sourceafis");
        Path cache = out.resolve("templates");
        Files.createDirectories(cache);
        List<String[]> rows = Files.readAllLines(root.resolve("out/index.csv")).stream()
            .skip(1).map(l -> l.split(",")).collect(Collectors.toList());
        int n = rows.size();
        FingerprintTemplate[] t = new FingerprintTemplate[n];

        long t0 = System.nanoTime();
        IntStream.range(0, n).parallel().forEach(i -> {
            String[] r = rows.get(i);
            String name = r[1] + "_" + r[2] + "_" + r[3];
            Path c = cache.resolve(name + ".cbor");
            try {
                if (Files.exists(c)) {
                    t[i] = new FingerprintTemplate(Files.readAllBytes(c));
                } else {
                    Path img = root.resolve("data/fvc/" + r[1] + "/" + r[2] + "_" + r[3] + ".tif");
                    t[i] = new FingerprintTemplate(new FingerprintImage(Files.readAllBytes(img),
                        new FingerprintImageOptions().dpi(dpi(r[1]))));
                    Files.write(c, t[i].toByteArray());
                }
            } catch (IOException e) { throw new UncheckedIOException(e); }
        });
        double tExtract = (System.nanoTime() - t0) / 1e9;

        long t1 = System.nanoTime();
        float[] m = new float[n * n];
        IntStream.range(0, n).parallel().forEach(i -> {
            FingerprintMatcher matcher = new FingerprintMatcher(t[i]);
            for (int j = 0; j < n; j++) m[i * n + j] = (float) matcher.match(t[j]);
        });
        double tMatch = (System.nanoTime() - t1) / 1e9;

        ByteBuffer buf = ByteBuffer.allocate(4 * n * n).order(ByteOrder.LITTLE_ENDIAN);
        for (float v : m) buf.putFloat(v);
        Files.write(out.resolve("matrix.f32"), buf.array());
        String json = String.format(Locale.ROOT,
            "{\"n\": %d, \"extract_seconds\": %.1f, \"match_seconds\": %.1f, \"comparisons\": %d, \"threads\": %d}%n",
            n, tExtract, tMatch, (long) n * n, java.util.concurrent.ForkJoinPool.getCommonPoolParallelism());
        Files.writeString(out.resolve("timing.json"), json);
        System.out.print(json);
    }
}
