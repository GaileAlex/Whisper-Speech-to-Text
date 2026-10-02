FROM eclipse-temurin:25-jdk-noble AS builder
WORKDIR /workspace
COPY .mvn/ .mvn
COPY mvnw pom.xml ./
RUN ./mvnw dependency:go-offline
COPY src ./src
RUN ./mvnw package -DskipTests

FROM eclipse-temurin:25-jre-noble

WORKDIR /app
COPY --from=builder /workspace/target/*.jar app.jar

RUN useradd --create-home spring
USER spring

EXPOSE 8389
ENTRYPOINT ["java", "-jar", "app.jar"]
